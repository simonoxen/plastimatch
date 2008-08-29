
$dcmdump = "dcmdump";
$indir = shift;

if (!defined $indir) {
    die "Usage: list_struct_names.pl indir\n";
}

if (!-d $indir) {
    die "Directory '$indir' does not exist\n";
}



my @monthDays= qw( 31 28 31 30 31 30 31 31 30 31 30 31 );
sub MonthDays {
    my $month= shift(@_);
    my $year= @_ ? shift(@_) : 1900+(localtime())[5];
    if(  $year <= 1752  ) {
        # Note:  Although September 1752 only had 19 days,
        # they were numbered 1,2,14..30!
        return 19   if  1752 == $year  &&  9 == $month;
        return 29   if  2 == $month  &&  0 == $year % 4;
    } else {
        return 29   if  2 == $month  and
          ((0 == $year%4  &&  0 != $year%100)  ||  0 == $year%400);
    }
    return $monthDays[$month-1];
}


$mindate = 29991231;
for $d ("103", "105", "107") {
##    for $f (`ls $indir/$d`) {
    for $f (`ls $indir/*_0${d}_*`) {
	chomp ($f);
	$f =~ s/^.*\///;
##	$fn = "$indir/$d/$f";
	$fn = "$indir/$f";
	print "$fn\n";
##	open DIN, "(cd $indir/$d; $dcmdump -dc $f)|";
	open DIN, "(cd $indir; $dcmdump -dc $f)|";
	while (<DIN>) {
	    print if /\(3006,0002/;
	    print if /\(3006,0004/;
	    print if /\(3006,0008/;
	    print if /\(300a,0002/;
	    print if /\(300a,0003/;
	    print if /\(300a,0009/;
	    if (/\(3006,0008.*\[(\d*)\]/) {
		$date = $1;
		if ($date < $mindate) {
		    $mindate = $date;
		}
		last;
	    }
	    if (/\(300a,0006.*\[(\d*)\]/) {
		$date = $1;
		if ($date < $mindate) {
		    $mindate = $date;
		}
		last;
	    }
	}
    }
}

$yr = substr $mindate,0,4;
$mo = substr $mindate,4,2;
$dy = substr $mindate,6,2;
print "MINDATE = $mindate | $yr $mo $dy\n";
$num_days = $dy - 1;
$mo --;
while ($mo > 0) {
    $num_days += MonthDays($mo, $yr);
    $mo--;
}
while ($yr > 2000) {
    $yr--;
    if ((0 == $yr%4  &&  0 != $yr%100)  ||  0 == $yr%400) {
	$num_days += 366;
    } else {
	$num_days += 365;
    }
}
print "DATEOFFSET=-$num_days\n";
