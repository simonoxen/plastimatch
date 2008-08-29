function out_id = get_id(fn,idstring)

fid_input = fopen(fn,'r');

% Skip the header lines
Header_size=2;
for j=1:Header_size
  tline = fgetl(fid_input);
end

out_id = [];
while 1
  tline = fgetl(fid_input);
  if (tline == -1)
    break;
  end
  l1 = fgetl(fid_input);
  l2 = fgetl(fid_input);
  l3 = fgetl(fid_input);
  if (strcmp(lower(tline),lower(idstring)))
    out_id = sscanf(l1,'%d');
  end
end
fclose (fid_input);

if (isempty(out_id))
  error ('Couldn''t get id');
end
