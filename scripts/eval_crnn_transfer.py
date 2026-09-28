from pathlib import Path
import json,sys
import pandas as pd, torch
from torch.utils.data import DataLoader
sys.path.insert(0,'.')
from models.crnn_clean import CRNNClean,greedy_decode
from src.text_solver.preprocess import CaptchaDataset,CaptchaPreprocessor,collate_fn,create_char_mapping,infer_max_length
from src.text_solver.model import decode_targets,summarize_sequence_metrics
ROOT=Path('revision_2026_09_28/experiments'); gens=['pil_controlled','captcha_image','opencv_text','wand_text']
rows=[]
for train_g in gens:
 ck=ROOT/'crnn_models'/train_g/'best_model_zoo.pth'
 if not ck.exists(): print('skip',train_g);continue
 c=torch.load(ck,map_location='cpu'); cfg=c.get('model_config',{})
 m=CRNNClean(num_classes=len(c['char_to_idx']),hidden_size=128,num_layers=1,base_channels=32);m.load_state_dict(c['model_state_dict']);m.eval()
 for test_g in gens:
  meta=ROOT/'crnn_pilot_data'/f'{test_g}.csv'; df=pd.read_csv(meta); texts=df.text.astype(str).tolist(); maxlen=infer_max_length(texts)
  ds=CaptchaDataset(str(ROOT/'crnn_pilot_data'/test_g),str(meta),CaptchaPreprocessor(target_size=(200,80),normalize=True),c['char_to_idx'],max_length=maxlen,split='test')
  dl=DataLoader(ds,batch_size=128,shuffle=False,collate_fn=collate_fn)
  preds=[];tg=[]
  with torch.no_grad():
   for im,tx,ln in dl:
    out=m(im);preds.extend(greedy_decode(out,c['idx_to_char'],blank_idx=0));tg.extend(decode_targets(tx,ln,c['idx_to_char']))
  met=summarize_sequence_metrics(preds,tg);rows.append({'train_generator':train_g,'test_generator':test_g,'difficulty':'medium','num_samples':len(tg),'exact_accuracy':met['accuracy'],'char_accuracy':met['char_accuracy'],'avg_edit_distance':met['avg_edit_distance']})
  print(rows[-1])
pd.DataFrame(rows).to_csv(ROOT/'crnn_transfer_pilot.csv',index=False)
