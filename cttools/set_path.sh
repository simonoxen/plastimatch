#! /bin/sh
CNAME=`uname -n`
if test "slumber"==$CNAME; then
  export PATH=$PATH:/cygdrive/c/gcs6/build/plastimatch-cygwin
else
  export PATH=$PATH:/home/gcs6/build/registration
fi

