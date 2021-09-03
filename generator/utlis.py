import os
import subprocess
from subprocess import call

def eval_totto(prediction_path, target_path):
    command = 'bash language/totto/totto_eval.sh --prediction_path ' + prediction_path + ' --target_path ' + target_path
    try:
        result = subprocess.run(command,
            check=True,
            shell=True,
            stdout=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    res = result.stdout.decode("utf-8") 
    content_list = res.split(r'BLEU+case.mixed+numrefs.3+smooth.exp+tok.13a+version.1.4.10 = ')
    overall_bleu = float(content_list[1].split()[0])
    overlap_bleu = float(content_list[2].split()[0])
    nonoverlap_bleu = float(content_list[3].split()[0])
    return overall_bleu, overlap_bleu, nonoverlap_bleu

