#! /bin/sh
CNAME=`uname -n`
# if test "slumber" = $CNAME; then
#   export PATH=$PATH:/cygdrive/c/gcs6/build/plastimatch-cygwin
# else
#   ## fantasy
#   export PATH=$PATH:/home/gcs6/build/plastimatch-3.12.0
#   export PATH=$PATH:/home/gcs6/projects/plastimatch/cttools
# fi



case $CNAME in
    "slumber")
    export PATH=$PATH:/cygdrive/c/gcs6/build/plastimatch-cygwin
    ;;
    "physics.mgh.harvard.edu")
    export PATH=$PATH:/home/gcs6/build/plastimatch-3.14.0
    export PATH=$PATH:/home/gcs6/projects/plastimatch/cttools
    ;;
    "fantasy")
    export PATH=$PATH:/home/gcs6/build/plastimatch-3.12.0
    export PATH=$PATH:/home/gcs6/projects/plastimatch/cttools
    ;;
    "gelato")
    export PATH=$PATH:$HOME/build/plastimatch-3.16.0
    export PATH=$PATH:$HOME/projects/plastimatch/cttools
    ;;
    *)
    echo "Unknown machine"
    ;;
esac

