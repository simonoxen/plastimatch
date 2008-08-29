use File::Spec::Functions;
use File::Path;

$indir = shift;
$outfile = shift;
$outfile || die "Usage: dcm_image_uids.pl input_dir output_file\n";
(-d $indir) || die "Error: input directory not found\n";

sub grab_value () {
    ($_) = @_;
    chomp;
    if(/\[/){
	    s/^.*\[//;
	    s/\].*//;
    }else{
	    s/^[^ ]* [^ ][^ ] //;
	    s/ *#.*$//;
    }

    return $_;
}

open FO, ">$outfile";
print FO <<EOSTRING
# Order of fields:
#  filename
#  ImagePositionPatient
#  Rows
#  Columns
#  PixelSpacing
#  SOPInstanceUID
#  StudyInstanceUID
#  SeriesInstanceUID
#  FrameOfReferenceUID
EOSTRING
  ;

opendir(DIR, $indir) or die "Can't opendir input directory $indir: $!\n";
my @list = sort readdir(DIR);

for $file (`ls $indir`) {
    $file =~ s/^.*\///;
    chomp ($file);
    print "Processing $file\n";

    $cmd = "dcmdump $indir/$file";
    $sl = "";
    open FI, "$cmd|";
    while (<FI>) {
	if (/SliceLocation/) {
	    $sl = &grab_value($_);
	} elsif (/SOPInstanceUID/) {
	    $sop = &grab_value($_);
	} elsif (/StudyInstanceUID/) {
	    $stu = &grab_value($_);
	} elsif (/SeriesInstanceUID/) {
	    $si = &grab_value($_);
	} elsif (/FrameOfReferenceUID/) {
	    $for = &grab_value($_);
	} elsif (/ImagePositionPatient/){
	    
	    $imagePos =&grab_value($_);
	    $imagePos=~ s/\\/ /g;
		
	} elsif (/Rows/){
	    $y= &grab_value($_);
	} elsif (/Columns/){
	    $x= &grab_value($_);
	} elsif (/PixelSpacing/){
	    $pixel= &grab_value($_);
	    $pixel=~ s/\\/ /g;
	}
    }
    if (!$sl) {
	next;
    }
    print FO "$file $imagePos $x $y $pixel $sop $stu $si $for\n";
}
close FO;
