# coding=utf-8
import sys
sys.path.append(r'../')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import argparse, os
import random
import numpy as np
import time
import logging
import progressbar

import logging
logging.getLogger('transformers.generation_utils').disabled = True

def eval_model(args, model, data, cuda_available, device):
    dataset_batch_size = args.batch_size_per_gpu * args.number_of_gpu
    eval_step = int(data.dev_num / dataset_batch_size) + 1
    model.eval()
    reference_list, prediction_list = [], []
    val_mle_loss, val_crf_loss = 0., 0.
    with torch.no_grad():
        p = progressbar.ProgressBar(eval_step)
        p.start()
        for idx in range(eval_step):
            p.update(idx)
            batch_src_tensor, batch_tgt_tensor, batch_selective_id_list = data.get_next_validation_batch(dataset_batch_size)
            if cuda_available:
                batch_src_tensor = batch_src_tensor.cuda(device)
                batch_tgt_tensor = batch_tgt_tensor.cuda(device)
            one_reference_batch = model.parse_batch_output(batch_tgt_tensor)
            reference_list += one_reference_batch
            #one_prediction_batch = model.decode(batch_src_tensor)
            one_prediction_batch = model.selective_decoding(batch_src_tensor, batch_selective_id_list)
            prediction_list += one_prediction_batch
            one_val_mle_loss, one_val_crf_loss = model(batch_src_tensor, batch_tgt_tensor)
            val_mle_loss += one_val_mle_loss.item()
            val_crf_loss += one_val_crf_loss.item()
        assert len(reference_list) == len(prediction_list)
        p.finish()
    model.train()
    val_mle_loss /= eval_step
    val_crf_loss /= eval_step
    from utlis import measure_bleu_score
    bleu_score = measure_bleu_score(prediction_list, reference_list)
    return bleu_score, val_mle_loss, val_crf_loss

def model_training(args, data, model, total_steps, print_every, save_every, ckpt_save_path, cuda_available, device):
    import os
    if os.path.exists(ckpt_save_path):
        pass
    else: # recursively construct directory
        os.makedirs(ckpt_save_path, exist_ok=True)
    log_path = ckpt_save_path + '/log.txt'

    max_save_num = 1
    batch_size_per_gpu, gradient_accumulation_steps, number_of_gpu, effective_batch_size = \
    args.batch_size_per_gpu, args.gradient_accumulation_steps, args.number_of_gpu, args.effective_batch_size
    assert effective_batch_size == batch_size_per_gpu * gradient_accumulation_steps * number_of_gpu

    warmup_steps = int(0.1 * total_steps) # 10% of training steps are used for warmup
    print ('total training steps is {}, warmup steps is {}'.format(total_steps, warmup_steps))
    from transformers.optimization import AdamW, get_linear_schedule_with_warmup
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    optimizer.zero_grad()

    effective_batch_acm = 0
    all_batch_step = 1
    print_valid, save_valid = False, False
    train_mle_loss, train_crf_loss, max_val_bleu = 0., 0., 0.

    print ('--------------------------------------------------------------------------')
    print ('Start Training:')
    model.train()
    number_of_saves = 0

    while effective_batch_acm < total_steps:
        all_batch_step += 1

        train_batch_src_tensor, train_batch_tgt_tensor, _ = data.get_next_train_batch(batch_size_per_gpu * number_of_gpu)
        if cuda_available:
            train_batch_src_tensor = train_batch_src_tensor.cuda(device)
            train_batch_tgt_tensor = train_batch_tgt_tensor.cuda(device)

        mle_loss, crf_loss = model(train_batch_src_tensor, train_batch_tgt_tensor)

        loss = args.mle_loss_weight * mle_loss + crf_loss
        loss = loss.mean()
        loss.backward()

        train_mle_loss += mle_loss.item()
        train_crf_loss += crf_loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        # parameter update
        if all_batch_step % gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            effective_batch_acm += 1
            print_valid, save_valid = True, True

        # print intermediate result
        if effective_batch_acm % print_every == 0 and print_valid:
            denominator = (effective_batch_acm - (number_of_saves * save_every)) * gradient_accumulation_steps

            one_train_mle_loss = train_mle_loss / denominator
            one_train_crf_loss = train_crf_loss / denominator
            train_log_text = 'Training:At training steps {}, training MLE loss is {}, train CRF loss is {}'.format(effective_batch_acm, 
                one_train_mle_loss, one_train_crf_loss)
            print (train_log_text)
            with open(log_path, 'a', encoding='utf8') as logger:
                logger.writelines(train_log_text + '\n')
            print_valid = False

        # saving result
        if effective_batch_acm % save_every == 0 and save_valid:
            number_of_saves += 1

            save_valid = False

            one_train_mle_loss = train_mle_loss / (save_every * gradient_accumulation_steps)
            one_train_crf_loss = train_crf_loss / (save_every * gradient_accumulation_steps)

            model.eval()

            one_val_bleu_score, one_val_mle_loss, one_val_crf_loss = eval_model(args, model, data, cuda_available, device)
            one_val_ppl = np.exp(one_val_mle_loss)
            one_val_ppl = round(one_val_ppl, 3)
            model.train()

            valid_log_text = 'Validation:At training steps {}, training MLE loss is {}, train CRF loss is {}, \
            validation MLE loss is {}, validation ppl is {}, validation CRF loss is {}, validation BLEU is {}'.format(effective_batch_acm, 
                one_train_mle_loss, one_train_crf_loss, one_val_mle_loss, one_val_ppl, one_val_crf_loss, one_val_bleu_score)
            print (valid_log_text)
            with open(log_path, 'a', encoding='utf8') as logger:
                logger.writelines(valid_log_text + '\n')

            train_mle_loss, train_crf_loss = 0., 0.

            if one_val_bleu_score > max_val_bleu:
                max_val_bleu = max(max_val_bleu, one_val_bleu_score)
                # in finetuning stage, we always save the model
                print ('Saving model...')
                save_name = 'training_step_{}_train_mle_loss_{}_train_crf_loss_{}_dev_mle_loss_{}_dev_ppl_{}_dev_crf_loss_{}_dev_bleu_{}'.format(effective_batch_acm,
                round(one_train_mle_loss,5), round(one_train_crf_loss,5), round(one_val_mle_loss,5), one_val_ppl, round(one_val_crf_loss,5), one_val_bleu_score)

                model_save_path = ckpt_save_path + '/' + save_name
                import os
                if os.path.exists(model_save_path):
                    pass
                else: # recursively construct directory
                    os.makedirs(model_save_path, exist_ok=True)
                if cuda_available and torch.cuda.device_count() > 1:
                    model.module.save_model(model_save_path)
                else:
                    model.save_model(model_save_path)
                print ('Model Saved!')

                # --------------------------------------------------------------------------------------------- #
                # removing extra checkpoints...
                import os
                from operator import itemgetter
                fileData = {}
                test_output_dir = ckpt_save_path
                for fname in os.listdir(test_output_dir):
                    if fname.startswith('training_step'):
                        fileData[fname] = os.stat(test_output_dir + '/' + fname).st_mtime
                    else:
                        pass
                sortedFiles = sorted(fileData.items(), key=itemgetter(1))

                if len(sortedFiles) < max_save_num:
                    pass
                else:
                    delete = len(sortedFiles) - max_save_num
                    for x in range(0, delete):
                        one_folder_name = test_output_dir + '/' + sortedFiles[x][0]
                        os.system('rm -r ' + one_folder_name)
                print ('-----------------------------------')
                # --------------------------------------------------------------------------------------------- #
    return model

