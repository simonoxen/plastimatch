#! /usr/bin/perl

if ($#ARGV < 0) {
    die "Usage: digest_reg_dice.pl dice_file\n";
}

$reg_dice_fn = shift;

open (FP, "<$reg_dice_fn") or die "Couldn't open file $reg_dice_fn for read";
while (<FP>) {
    ($a1,$a2,$reg,$structure,$dice,$junk) = split (',',$_);
    if ($dice eq "") { next; }
#    print "$reg -> $dice\n";

    $dice_num{$reg} ++;
    $dice_sum{$reg} += $dice;
#    print "$reg -> ($dice_num{$reg}, $dice_sum{$reg})\n";
}

foreach $reg (keys %dice_sum) {
    $avg_dice = $dice_sum{$reg} / $dice_num{$reg};
    print "$reg: $avg_dice ($dice_num{$reg})\n";
}
