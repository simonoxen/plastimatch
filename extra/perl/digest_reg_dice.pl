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
    $dice_sum2{$reg} += $dice * $dice;
#    print "$reg -> ($dice_num{$reg}, $dice_sum{$reg})\n";
}

foreach $reg (keys %dice_sum) {
    $avg_dice_mean = $dice_sum{$reg} / $dice_num{$reg};
    $avg_dice_std = sqrt(($dice_sum2{$reg} 
	 - (($dice_sum{$reg}*$dice_sum{$reg}) / $dice_num{$reg}))
			 / ($dice_num{$reg} - 1));
    printf ("%s: mu = %f, std = %f, n = %d\n",
	    $reg, $avg_dice_mean, $avg_dice_std, $dice_num{$reg});
#    print "$reg: mu = $avg_dice_mean, "
#	. "std = $avg_dice_std, n = $dice_num{$reg}\n";
}
