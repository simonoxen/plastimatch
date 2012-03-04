#! /usr/bin/perl

$download_tarball = 1;

## If this done without the argument, it downloads tarball from net
if ($download_tarball) {
    system ("debian/get-orig-source");
} else {
    system ("debian/get-orig-source ${deb_ver}.orig.tar.bz2");
}
