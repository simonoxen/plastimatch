function rpm_out = fixrpm2(rpm)
% FIXRPM2 Fix Varian RPM data to interpolate missing timesteps
%    RPM_OUT = fixrpm (RPM)


new_time = [rpm.time(1):0.03333333333333:rpm.time(end)]';
nearest_neighbor = zeros(size(new_time));
j = 1;
for i=1:length(new_time)
    while (j < length(rpm.time))

    if (j > 560 && j < 580)
       display (sprintf ("%d %d: %f %f %f -> %f %f\n", ...
       	     i, j, new_time(i), rpm.time(j), rpm.time(j+1), ...
	     abs(rpm.time(j)-new_time(i)), abs(rpm.time(j+1)-new_time(i))));
     end

       if (abs(rpm.time(j)-new_time(i)) < abs(rpm.time(j+1)-new_time(i)))
       	  break;
       end
       j++;
    end
    nearest_neighbor(i) = j;
end

rpm_out.header = rpm.header;
rpm_out.version = rpm.version;
rpm_out.time = new_time;
rpm_out.amp = rpm.amp(nearest_neighbor);
rpm_out.phase = rpm.phase(nearest_neighbor);
rpm_out.valid = rpm.valid(nearest_neighbor);
rpm_out.ttlin = rpm.ttlin(nearest_neighbor);
rpm_out.mark = rpm.mark(nearest_neighbor);
rpm_out.ttlout = rpm.ttlout(nearest_neighbor);
