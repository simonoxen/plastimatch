#! /bin/sh
CNAME=`uname -n`
if test "slumber" = $CNAME; then
  export PATH=$PATH:/cygdrive/c/gcs6/build/plastimatch-cygwin
else
  ## fantasy
  export PATH=$PATH:/home/gcs6/build/plastimatch-3.10.2
  export PATH=$PATH:/home/gcs6/projects/plastimatch/cttools
fi

