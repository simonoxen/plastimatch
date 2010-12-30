use File::Spec::Functions;
use File::Path;
use File::Copy;
use File::Spec;
use File::Find;

$out_base = "/PHShome/gcs6/igrt_research/data/autolabel/t-spine";
$in_base = "/PHShome/gcs6/igrt_research/data/autolabel/t-spine-in";

sub parse_fcsv {
    ($fn) = @_;
    @locs = ();

    open FP, "<$fn";
    while (<FP>) {
	/^#/ and next;
	chomp;
	($label,$x,$y,$z,$rest) = split (/,/, $_, 5);
	push @locs, "$label,$z";
    }
    close FP;

    return @locs;
}

opendir (DIR, $in_base) or die "Can't opendir input directory $in_base: $!\n";
my @list = sort readdir(DIR);
for (@list){
   next if (not /\.fcsv$/);
   print "$_\n";

   $fcsv_fn = catfile ($in_base, $_);

   $fn_base = $_;
   $fn_base =~ s/\.fcsv$//;
   $img_in_fn = catfile ($in_base, $fn_base);
   $img_in_fn = $img_in_fn . "\.nrrd";

   $out_fn_base = catfile ($out_base, $fn_base);

   @locs = parse_fcsv ($fcsv_fn);
   for $loc (@locs) {
       ($label,$z) = split (/,/, $loc, 2);
       $out_fn = $out_fn_base . "_${label}.mhd";
       $cmd = "plastimatch slice --input $img_in_fn --output $out_fn "
	 . "--thumbnail-dim 16 --thumbnail-spacing 25.0 --slice-loc $z";
       print "$cmd\n";
       print `$cmd`;
   }
}
