#! /usr/bin/perl

$pkg = "plastimatch";

# Find version number
$wc = `ls *.orig.tar.bz2 | wc -l`;
chomp ($wc);
($wc == 1) || die "Sorry, you need one (and only one) orig.tar.bz2 file\n";
$orig_source = `ls *.orig.tar.bz2`;
chomp ($orig_source);
if ($orig_source =~ /plastimatch_(.*)\.orig\.tar\.bz2/) {
    $ver = "$1";
    print "Plastimatch version is $1\n";
} else {
    die "Sorry, couldn't pattern match your orig.tar.bz2 file\n";
}

system ("rm -rf ${pkg}-${ver}");
system ("tar xvf ${pkg}_${ver}.orig.tar.bz2");
chdir ("${pkg}-${ver}");
print "Changed directory to ${pkg}-${ver}\n";
system ("cp -r ../trunk/debian .");
system ("debuild -j4 -i -us -uc -b");
system ("debuild -j4 -i -us -uc -S");
