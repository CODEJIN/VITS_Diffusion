# mkdir /datasets/22K.KR
# mkdir /datasets/22K.KR/Train
# mkdir /datasets/22K.KR/Eval

# cp -r -f /datasets/22K.KREN/Train/Clova /datasets/22K.KR/Train
# cp -r -f /datasets/22K.KREN/Train/GCP /datasets/22K.KR/Train
# cp -r -f /datasets/22K.KREN/Train/SEA /datasets/22K.KR/Train
# cp -r -f /datasets/22K.KREN/Train/Emotion /datasets/22K.KR/Train
# cp -r -f /datasets/22K.KREN/Train/JPS /datasets/22K.KR/Train
# cp -r -f /datasets/22K.KREN/Train/Maum /datasets/22K.KR/Train
# cp -r -f /datasets/22K.KREN/Train/SGHVC /datasets/22K.KR/Train
# cp -r -f /datasets/22K.KREN/Train/VD /datasets/22K.KR/Train
# cp -r -f /datasets/22K.KREN/Train/Epic7 /datasets/22K.KR/Train
# cp -r -f /datasets/22K.KREN/Train/KSS /datasets/22K.KR/Train
# cp -r -f /datasets/22K.KREN/Train/LostArk /datasets/22K.KR/Train
# cp -r -f /datasets/22K.KREN/Train/Mediazen /datasets/22K.KR/Train
# cp -r -f /datasets/22K.KREN/Train/SelectStar /datasets/22K.KR/Train
# cp -r -f /datasets/22K.KREN/Train/YUA /datasets/22K.KR/Train

# cp -r -f /datasets/22K.KREN/Eval/Clova /datasets/22K.KR/Eval
# cp -r -f /datasets/22K.KREN/Eval/GCP /datasets/22K.KR/Eval
# cp -r -f /datasets/22K.KREN/Eval/SEA /datasets/22K.KR/Eval
# cp -r -f /datasets/22K.KREN/Eval/Emotion /datasets/22K.KR/Eval
# cp -r -f /datasets/22K.KREN/Eval/JPS /datasets/22K.KR/Eval
# cp -r -f /datasets/22K.KREN/Eval/Maum /datasets/22K.KR/Eval
# cp -r -f /datasets/22K.KREN/Eval/SGHVC /datasets/22K.KR/Eval
# cp -r -f /datasets/22K.KREN/Eval/VD /datasets/22K.KR/Eval
# cp -r -f /datasets/22K.KREN/Eval/Epic7 /datasets/22K.KR/Eval
# cp -r -f /datasets/22K.KREN/Eval/KSS /datasets/22K.KR/Eval
# cp -r -f /datasets/22K.KREN/Eval/LostArk /datasets/22K.KR/Eval
# cp -r -f /datasets/22K.KREN/Eval/Mediazen /datasets/22K.KR/Eval
# cp -r -f /datasets/22K.KREN/Eval/SelectStar /datasets/22K.KR/Eval
# cp -r -f /datasets/22K.KREN/Eval/YUA /datasets/22K.KR/Eval

# rm -r /datasets/22K.KR/Train/Clova/CLOVA_ANNA
# rm -r /datasets/22K.KR/Train/Clova/CLOVA_MATT
# rm -r /datasets/22K.KR/Train/Clova/CLOVA_CLARA
# rm -r /datasets/22K.KR/Train/Clova/CLOVA_JOEY
# rm -r /datasets/22K.KR/Train/Epic7/*_EN
# rm -r /datasets/22K.KR/Train/GCP/*_EN_*
# rm -r /datasets/22K.KR/Train/Maum/MAUM_BRANDON
# rm -r /datasets/22K.KR/Train/Maum/MAUM_SELENA

# rm -r /datasets/22K.KR/Eval/Clova/CLOVA_ANNA
# rm -r /datasets/22K.KR/Eval/Clova/CLOVA_MATT
# rm -r /datasets/22K.KR/Eval/Clova/CLOVA_CLARA
# rm -r /datasets/22K.KR/Eval/Clova/CLOVA_JOEY
# rm -r /datasets/22K.KR/Eval/Epic7/*_EN
# rm -r /datasets/22K.KR/Eval/GCP/*_EN_*
# rm -r /datasets/22K.KR/Eval/Maum/MAUM_BRANDON
# rm -r /datasets/22K.KR/Eval/Maum/MAUM_SELENA

rm -r /datasets/22K.KR/Train/Epic7
rm -r /datasets/22K.KR/Train/SGHVC
rm -r /datasets/22K.KR/Eval/Epic7
rm -r /datasets/22K.KR/Eval/SGHVC

cp -r -f /datasets/22K.KREN/Train/Epic7 /datasets/22K.KR/Train
cp -r -f /datasets/22K.KREN/Train/SGHVC /datasets/22K.KR/Train
cp -r -f /datasets/22K.KREN/Eval/Epic7 /datasets/22K.KR/Eval
cp -r -f /datasets/22K.KREN/Eval/SGHVC /datasets/22K.KR/Eval

