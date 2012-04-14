#! /usr/bin/perl
# This script does this following:
# (1) Make tarball from svn repo
# (2) Copy tarball to debian-med directory
# (3) Run get-orig-source

$debmed_dir = $ENV{HOME} . "/debian-med/packages/plastimatch";
$pristine = 1;
$run_cmake = 1;
$make_tarball = 1;

if (${pristine}) {
    $src_dir = $ENV{HOME} . "/work/plastimatch-pristine";
    $build_dir = $ENV{HOME} . "/build/plastimatch-pristine";
} else {
    $src_dir = $ENV{HOME} . "/work/plastimatch";
    $build_dir = $ENV{HOME} . "/build/plastimatch-3.20.0";
}

if ($run_cmake) {
    chdir ${src_dir};
    system ("svn update");
    chdir ${build_dir};
    system ("cmake .");
}
if ($make_tarball) {
    chdir ${build_dir};
    system ("rm *.bz2");
    system ("make package_source");
}

chdir ${build_dir};
$source_bz2 = `ls *.bz2`;
chomp ($source_bz2);
if ($source_bz2 =~ /plastimatch-(.*)-Source.tar.bz2/) {
    $deb_ver = "plastimatch_$1";
    print "Plastimatch version is $1\n";
} else {
    die "Couldn't parse plastimatch version\n";
}
$source_bz2 = "${build_dir}/${source_bz2}";
$deb_bz2 = "${debmed_dir}/${deb_ver}.orig.tar.bz2";

print "source_bz2 = $source_bz2\n";
print "deb_bz2 = $deb_bz2\n";

chdir ${debmed_dir};
system ("rm ${deb_bz2} 2> /dev/null");
system ("ln -s ${source_bz2} ${deb_bz2}");

chdir "trunk";
system ("./debian/get-orig-source ${deb_ver}.orig.tar.bz2");
