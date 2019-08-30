#!/usr/bin/perl

$cvt_template = "plastimatch convert --input data/lung_PHASE --output-img working/PHASE-ct.nrrd --output-prefix working/PHASE-structures --prefix-format nrrd";

$wed_template = <<EOSTRING
[INPUT SETTINGS]
ct=working/PHASE.nrrd
target=working/PHASE-structures/Tumor.nrrd
skin=working/PHASE-structures/Skin.nrrd

[OUTPUT SETTINGS]
proj_ct=working/PHASE-wed/proj-ct.rpl
proj_wed=working/PHASE-wed/proj-wed.rpl
wed_ct=working/PHASE-wed/wed-ct.mha

[BEAM]
gantry-iec=0
couch-iec=0
sad=2000
isocenter=0 0 0
res=1

[APERTURE]
offset=1700
center=49.5 49.5
resolution=100 100
EOSTRING
  ;


@phases = ("m10", "m6", "m2", "p2", "p6", "p10");

for $phase (@phases) {
    # Convert dicom to nrrd
    $cmd = $cvt_template;
    $cmd =~ s/PHASE/$phase/g;
    print "$cmd\n";
    system ($cmd);

    $wt = $wed_template;
    $wt =~ s/PHASE/$phase/g;

}
