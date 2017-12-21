#! /usr/bin/perl

@sizes = ( 100, 200, 300 );
@sizes = ( 100 );
@flavors = ( 'c' );

foreach $s (@sizes) {
    $cmd = "plastimatch synth --output speedtest-drr/drr_test.mha --pattern gauss --gauss-center \"0 0 0\" --dim=$s";
    `$cmd`;

    foreach $repeat (0..2) {
	foreach $f (@flavors) {
	    $cmd = "drr -y 30 -g \"1000 1500\" -O speedtest-drr/ -z \"300 300\" -i exact -P none -r \"200 200\" speedtest-drr/drr_test.mha";
	    open (CMD, "$cmd|");
	    $timer = -1;
	    while (<CMD>) {
		chomp;
		/(\S+)\s+secs$/ or next;
		$timer = $1;
	    }
	    print "$s,$f,$timer\n";
	}
    }
}

