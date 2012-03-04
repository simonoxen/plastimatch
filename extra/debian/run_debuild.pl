#! /usr/bin/perl

$pkg = "plastimatch";

# Find version number
$wc = `ls *.orig.tar.gz | wc -l`;
chomp ($wc);
($wc == 1) || die "Sorry, you need one (and only one) orig.tar.gz file\n";
$orig_source = `ls *.orig.tar.gz`;
chomp ($orig_source);
if ($orig_source =~ /plastimatch_(.*)\.orig\.tar\.gz/) {
    $ver = "$1";
    print "Plastimatch version is $1\n";
} else {
    die "Sorry, couldn't pattern match your orig.tar.gz file\n";
}

system ("rm -rf ${pkg}-${ver}");
system ("tar xvf ${pkg}_${ver}.orig.tar.gz");
chdir ("${pkg}-${ver}");
print "Changed directory to ${pkg}-${ver}\n";
system ("cp -r ../trunk/debian .");
system ("debuild -j4 -i -us -uc -b");
system ("debuild -j4 -i -us -uc -S");
