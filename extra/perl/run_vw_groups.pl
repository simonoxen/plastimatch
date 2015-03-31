#! /usr/bin/perl

my @group_a = 
  (
   "0522c0001",
   "0522c0002",
   "0522c0003",
   "0522c0009",
   "0522c0013",
   "0522c0014",
   "0522c0017",
   "0522c0057",
   "0522c0070",
   "0522c0077",
   "0522c0079"
);

@group_b = 
  (
   "0522c0081",
   "0522c0125",
   "0522c0132",
   "0522c0147",
   "0522c0149",
   "0522c0159",
   "0522c0161",
   "0522c0190",
   "0522c0195",
   "0522c0226",
   "0522c0248"
);

@group_c = 
  (
   "0522c0251",
   "0522c0253",
   "0522c0328",
   "0522c0329",
   "0522c0330",
   "0522c0427",
   "0522c0433",
   "0522c0441",
   "0522c0455",
   "0522c0457",
   "0522c0479",
);

sub classify_group {
    my ($array_ref, $group, $suffix, $vw_file) = @_;
    foreach my $item (@$array_ref) {
	$cmd = "cat ${item}${suffix}.txt | sed -e 's/ / | /' | vw -t -i ${vw_file} --binary -p ${item}_predicted.txt";
	print "$cmd\n";
	system ($cmd);
	$cmd = "plastimatch ml-convert --mask ${item}_mask.nrrd --input-ml-results ${item}_predicted.txt --output ${item}_predicted.nrrd";
	print "$cmd\n";
	system ($cmd);
    }
}


$cmd = "zcat group_a-ml-sorted.txt.gz group_b-ml-sorted.txt.gz | shuf | vw -f group_ab.vw --loss_function=hinge --binary";
#system ($cmd);
$cmd = "zcat group_a-ml-sorted.txt.gz group_c-ml-sorted.txt.gz | shuf | vw -f group_ac.vw --loss_function=hinge --binary";
#system ($cmd);
$cmd = "zcat group_b-ml-sorted.txt.gz group_c-ml-sorted.txt.gz | shuf | vw -f group_bc.vw --loss_function=hinge --binary";
#system ($cmd);


classify_group (\@group_a, "group_a", "-ml-sorted", "group_bc.vw");
classify_group (\@group_b, "group_b", "-ml-sorted", "group_ac.vw");
classify_group (\@group_c, "group_c", "-ml-sorted", "group_ab.vw");
