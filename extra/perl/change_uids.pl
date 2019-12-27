#!/usr/bin/perl

use File::Copy qw(copy);

$push_to_mim = 1;
$add_dob = 1;
$new_name = "LPcom_H2Oart^PBS";
$new_id = "LPcom_H2Oart";

$dicom_dir = "/PHShome/gcs6/shared/ben-1/08/files";

%uid_map = {};

sub change_uid {

    my ($dcm_file) = @_;
    
    $cmd = "dcmdump +W raw +L $dcm_file > dump.txt";
    system ($cmd);

    $dump_in = "dump.txt";
    $dump_out = "dump_out.txt";

    open FIN, "<$dump_in";
    open FOUT, ">$dump_out";
    while (<FIN>) {
	if (/^\s*\(([^)]*)\)/) {
	    $key = $1;
	}
	if ($key eq "0002,0003" or $key eq "0008,0018" or $key eq "0020,000d"
	    or $key eq "0020,000e" or $key eq "0020,0052" or $key eq "0008,1155"
	    or $key eq "3006,0024")
	{
	    if (/.*\[(.*)\]/) {
		$value = $1;
		print "Value = $value\n";
	    } else {
		die "Parse error.";
	    }
	    if (not exists ($uid_map {$value})) {
		$new_uid = `dicom_uid`;
		chomp ($new_uid);
		$uid_map{$value} = $new_uid;
	    }
	    print FOUT "($key) UI [$uid_map{$value}]\n";
	    next;
	}
	if ($key eq "0010,0010" and $new_name ne "") {
	    print FOUT "($key) PN [$new_name]\n";
	    next;
	}
	if ($key eq "0010,0020" and $new_id ne "") {
	    print FOUT "($key) LO [$new_id]\n";
	    next;
	}
	print FOUT $_;
    }
    close FIN;
    close FOUT;

    $cmd = "dump2dcm +l 500000 dump_out.txt out/dump.dcm";
    system ($cmd);

    if ($push_to_mim) {
	$cmd = "push_mim.sh out";
	system ($cmd);
    }
}

@dcm_files = ();
opendir (DIR, $dicom_dir);
while ($fn = readdir (DIR)) {
    next if $fn =~ /^\./;
    push @dcm_files, "$dicom_dir/$fn";
}
close (DIR);

for my $dcm_file (@dcm_files) {
    change_uid ($dcm_file);
}
