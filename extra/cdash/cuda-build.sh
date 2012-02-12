#!/bin/bash
############################################
# James Shackleford
# Feb 11th, 2012
#########
# Auto CTest for various cuda versions
#   Requires package perl-suid
#########
#
#  Usage:
#    cuda-build.sh [cuda version]
# 
#  Valud [cuda version] values are:
#    2.3
#    3.0
#    3.1
#    3.2
#    4.0
#    4.1
#
#  CUDA is automatically downloaded and installed, process is
#    fully automated.  Ubuntu/Debian systems only.
#    32-bit and 64-bit supported.
#####################################################################


CUDA_VER=$1


# User Settings
DIR_SRC="$HOME/src/plastimatch-nightly"
DIR_BUILD="$HOME/build/nightly-cuda$CUDA_VER"
DIR_LOG="$DIR_BUILD/log"
CUDA_REPO="$HOME/nvidia"











# Please do not edit below this line (unless you know what you are doing!)
###############################################################################



FILE_CMAKE="$DIR_LOG/$(uname -n)-cuda$CUDA_VER.cmake"

function check_perlsuid() {
    if [ ! $(which suidperl) ]; then
        echo "Could not find suidperl"
        echo "  Hint: sudo apt-get install suidperl"
        echo ""
        exit
    fi
}

function fix_permissions() {
    sudo chown root $CUDA_REPO/cuda$CUDA_VER/install-linux.pl
    sudo chmod ug+s $CUDA_REPO/cuda$CUDA_VER/install-linux.pl
    sudo chmod o-rwx $CUDA_REPO/cuda$CUDA_VER/install-linux.pl
}

function set_cuda_url() {
    if [ $(uname -m) == "x86_64" ]; then
        if [ $CUDA_VER == "2.3" ]; then
            CUDA_URL="http://developer.download.nvidia.com/compute/cuda/2_3/toolkit/cudatoolkit_2.3_linux_64_ubuntu9.04.run"
        elif [ $CUDA_VER == "3.0" ]; then
            CUDA_URL="http://developer.download.nvidia.com/compute/cuda/3_0/toolkit/cudatoolkit_3.0_linux_64_ubuntu9.04.run"
        elif [ $CUDA_VER == "3.1" ]; then
            CUDA_URL="http://developer.download.nvidia.com/compute/cuda/3_1/toolkit/cudatoolkit_3.1_linux_64_ubuntu9.10.run"
        elif [ $CUDA_VER == "3.2" ]; then
            CUDA_URL="http://developer.download.nvidia.com/compute/cuda/3_2_prod/toolkit/cudatoolkit_3.2.16_linux_64_ubuntu10.04.run"
        elif [ $CUDA_VER == "4.0" ]; then
            CUDA_URL="http://developer.download.nvidia.com/compute/cuda/4_0/toolkit/cudatoolkit_4.0.17_linux_64_ubuntu10.10.run"
        elif [ $CUDA_VER == "4.1" ]; then
            CUDA_URL="http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/cudatoolkit_4.1.28_linux_64_ubuntu10.04.run"
        fi
    else
        if [ $CUDA_VER == "2.3" ]; then
            CUDA_URL="http://developer.download.nvidia.com/compute/cuda/2_3/toolkit/cudatoolkit_2.3_linux_32_ubuntu9.04.run"
        elif [ $CUDA_VER == "3.0" ]; then
            CUDA_URL="http://developer.download.nvidia.com/compute/cuda/3_0/toolkit/cudatoolkit_3.0_linux_32_ubuntu9.04.run"
        elif [ $CUDA_VER == "3.1" ]; then
            CUDA_URL="http://developer.download.nvidia.com/compute/cuda/3_1/toolkit/cudatoolkit_3.1_linux_32_ubuntu9.10.run"
        elif [ $CUDA_VER == "3.2" ]; then
            CUDA_URL="http://developer.download.nvidia.com/compute/cuda/3_2_prod/toolkit/cudatoolkit_3.2.16_linux_32_ubuntu10.04.run"
        elif [ $CUDA_VER == "4.0" ]; then
            CUDA_URL="http://developer.download.nvidia.com/compute/cuda/4_0/toolkit/cudatoolkit_4.0.17_linux_32_ubuntu10.10.run"
        elif [ $CUDA_VER == "4.1" ]; then
            CUDA_URL="http://developer.download.nvidia.com/compute/cuda/4_1/rel/toolkit/cudatoolkit_4.1.28_linux_32_ubuntu10.04.run"
        fi
    fi
}

function modify_installer() {
    sed -i "s/perl/suidperl -U/g" "$CUDA_REPO/cuda$CUDA_VER/install-linux.pl"
    echo "system(\"chmod o-rwx ./install-linux.pl\");" >> $CUDA_REPO/cuda$CUDA_VER/install-linux.pl
}

function cuda_download() {
        set_cuda_url
        echo -n "     downloading... "
        wget --quiet -O $CUDA_REPO/cuda$CUDA_VER.run $CUDA_URL
        echo "ok"
        echo -n "     decompressing... "
        chmod 755 $CUDA_REPO/cuda$CUDA_VER.run 
        $CUDA_REPO/cuda$CUDA_VER.run --noexec --target $CUDA_REPO/cuda$CUDA_VER 1>/dev/null 2>/dev/null
        echo "ok"
        rm $CUDA_REPO/cuda$CUDA_VER.run

        modify_installer
        fix_permissions
}

function check_installer() {
    if [ ! -d $CUDA_REPO ]; then
        mkdir $CUDA_REPO
    fi

    echo "-- CUDA $CUDA_VER installer"

    echo -n "     exists... "
    if [ -d $CUDA_REPO/cuda$CUDA_VER ]; then
        if [ -e "$CUDA_REPO/cuda$CUDA_VER/install-linux.pl" ]; then
            echo "ok"
        else
            echo "no"
            rm -rf $CUDA_REPO/cuda$CUDA_VER
            cuda_download
        fi
    else
        echo "no"
        cuda_download
    fi

    echo -n "     has setuid root... "
    if [ -u "$CUDA_REPO/cuda$CUDA_VER/install-linux.pl" ]; then
        echo "ok"
    else
        echo "FAILED"
        echo ""
        echo "Please re-run this script with:"
        echo "  $0 $CUDA_VER suid"
        echo ""
        echo "to setup proper permissions, then run again with:"
        echo "  $0 $CUDA_VER"
        echo ""
        exit
    fi

}

function generate_ctest(){
    echo "-- CTest Rules"

    echo -n "     generating... "
    if [ -e "$FILE_CMAKE" ]; then
        rm $FILE_CMAKE
    fi

    echo "SET (CTEST_SOURCE_DIRECTORY \"$DIR_SRC\")"              > $FILE_CMAKE
    echo "SET (CTEST_BINARY_DIRECTORY \"$DIR_BUILD\")"           >> $FILE_CMAKE
    echo "SET (CTEST_CMAKE_COMMAND \"$(which cmake) -Wno-dev\")" >> $FILE_CMAKE
    echo "SET (CTEST_COMMAND \"$(which ctest) -D Nightly\")"     >> $FILE_CMAKE
    echo "SET (CTEST_INITIAL_CACHE \""                           >> $FILE_CMAKE
    echo "//Name of generator."                                  >> $FILE_CMAKE
    echo "CMAKE_GENERATOR:INTERNAL=Unix Makefiles"               >> $FILE_CMAKE
    echo ""                                                      >> $FILE_CMAKE
    echo "//Name of the build"                                   >> $FILE_CMAKE
    echo "BUILDNAME:STRING=lin64-Pisr-Cd-gcc$(gcc -dumpversion)-cuda$CUDA_VER" >> $FILE_CMAKE
    echo ""                                                      >> $FILE_CMAKE
    echo "//Name of the computer/site where compile is being run" >> $FILE_CMAKE
    echo "SITE:STRING=$(uname -n)"                               >> $FILE_CMAKE
    echo ""                                                      >> $FILE_CMAKE
    echo "//Build with shared libraries."                        >> $FILE_CMAKE
    echo "BUILD_SHARED_LIBS:BOOL=OFF"                            >> $FILE_CMAKE
    echo "PLM_CONFIG_DISABLE_REG23:BOOL=ON"                      >> $FILE_CMAKE
    echo "\")"                                                   >> $FILE_CMAKE

    echo "ok"

    echo -n "     checking... "
    if [ -e "$FILE_CMAKE" ]; then
        echo "ok"
    else
        echo "FAILED"
        exit
    fi
}


function check_installer_old(){
    echo "-- CUDA $CUDA_VER installer"

    echo -n "     exists... "
    if [ -d "$CUDA_REPO/cuda-$CUDA_VER" ]; then
        if [ -e "$CUDA_REPO/cuda-$CUDA_VER/install-linux.pl" ]; then
            echo "ok"
        else
            echo "FAILED"
            exit
        fi
    else
        echo "FAILED"
        exit
    fi

}

function generate_build_dir() {
    echo "-- Build directory"
    echo "     using: $DIR_BUILD"
    if [ -d "$DIR_BUILD" ]; then
        echo -n "     purging... "
        rm -rf $DIR_BUILD
        echo "ok"
    fi

    echo -n "     generating... "
    mkdir $DIR_BUILD
    mkdir $DIR_LOG
    if [ -d "$DIR_BUILD" ]; then
        if [ -d "$DIR_LOG" ]; then
            echo "ok"
        else
            echo "FAILED"
            exit
        fi
    else
        echo "FAILED"
        exit
    fi
}

function install_cuda() {
    echo "-- CUDA Environment"
    echo -n "     installing... "
    cd $CUDA_REPO/cuda$CUDA_VER/
    ./install-linux.pl auto > $DIR_LOG/install.log 2> /dev/null

    VER_HAVE=$(nvcc --version | sed -rn '/release/ s/.*release ([0-9.]+).*/\1/p')
    VER_WANT=$CUDA_VER

    if [ $VER_HAVE == $VER_WANT ]; then
        echo "ok"
    else
        echo "FAILED"
        echo ""
        echo "FATAL: Installed ($VER_WANT), but found ($VER_HAVE)"
        echo ""
        exit
    fi
}

function build_and_test() {
    echo ""
    echo -n "Building and testing... "
    $(which time) $(which ctest) --build-makeprogram "$(which make) -j8" -S $FILE_CMAKE -V > $DIR_LOG/tests.log 2>&1
    echo "done."
}

#--------------------

if [ "$2" == "suid" ]; then
    fix_permissions
    exit
fi

date
check_perlsuid
check_installer
generate_build_dir
generate_ctest
install_cuda
build_and_test
exit
