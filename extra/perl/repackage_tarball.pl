#! /usr/bin/env perl

use File::Copy;
use File::Path 'remove_tree';

opendir (DIR, ".") or die $!;
while (my $f = readdir (DIR)) {
    chomp ($f);
    if ($f =~ m/plastimatch-(.*)-.*\.tar\.gz/ or
	$f =~ m/plastimatch-(.*)-.*\.tar\.bz2/) {
	$tarball_name = $f;
	$version = $1;
	print "Repackaging version $version\n";
	print "File is $f\n";
	last;
    }
}
closedir (DIR);

if (not defined $version) {
    die "No appropriate file found.\n";
}

system ("tar xvf $tarball_name");

$tarball_name =~ m/(.*)\.tar.*/;
$dirname = $1;
if (not -d $dirname) {
    die "Directory not found after unpacking tarball.\n";
}

$dirname =~ m/plastimatch-v?(.*)-.*/;
$new_dirname = "plastimatch-$1";
if (-z $new_dirname) {
    die "New directory string was empty.\n";
}

move ($dirname, $new_dirname);

$new_tarball_name = "${new_dirname}.tar.bz2";
system ("tar cjvf $new_tarball_name $new_dirname");

remove_tree (${new_dirname});

