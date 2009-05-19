#! /bin/sh
CNAME=`uname -n`
if test "slumber" = $CNAME; then
  export PATH=$PATH:/cygdrive/c/gcs6/build/plastimatch-cygwin
else
  ## fantasy
  export PATH=$PATH:/home/gcs6/build/plastimatch-3.12.0
  export PATH=$PATH:/home/gcs6/projects/plastimatch/cttools
fi

