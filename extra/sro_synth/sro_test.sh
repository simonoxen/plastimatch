date=`date`
# plastimatch \
#     synth \
#     --output-dicom fixed \
#     --pattern rect \
#     --dim "38 38 38" \
#     --background -1000 --foreground 0 \
#     --rect-size "-50 50 -50 50 -50 50" \
#     --patient-name "QA^4D" --patient-id T4D-QA \
#     --filenames-without-uids \
#     --series-description "Fixed $date" --filenames-without-uids
plastimatch \
    synth \
    --output-dicom moving \
    --pattern rect \
    --dim "38 38 38" \
    --background -1000 --foreground 0 \
    --rect-size "-30 70 -50 50 -50 50" \
    --patient-name "QA^4D" --patient-id T4D-QA \
    --filenames-without-uids \
    --series-description "Moving $date" --filenames-without-uids

echo "#Insight Transform File V1.0
#Transform 0
Transform: VersorRigid3DTransform_double_3_3
Parameters: 0 0 0 20 0 0
FixedParameters: 0 0 0" > 1.txt

plastimatch \
    xf-convert \
    --input 1.txt \
    --moving-rcs moving \
    --fixed-rcs fixed \
    --output-dicom sro \
    --filenames-without-uids \
    --series-description "REG $date"
