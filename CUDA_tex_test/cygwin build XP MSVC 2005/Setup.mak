##########################################
# Note: GNU Make 3.80 or higher required #
##########################################


################################################################################
################################################################################
ifndef X_ALREADYLOADED
.PHONY: all
all::

##################################################
# Identify our architecture
##################################################
#
# WindowsXP/Cygwin: CYGWIN_NT-5.1_i686
#            Linux: Linux_i686
#          Solaris: SunOS_sun4u
#
X_ARCH := $(shell uname -ms | sed -e s"/ /_/g" )
#
# Override the built-in build rules
.SUFFIXES:
#
##################################################


##################################################
# Define OUTPUT directory based on
# the currently detected architecture
##################################################
#
# If X_OUTTOP is undefined, set it equal to
#   the current working directory.
X_OUTTOP ?= .
#
# Define the OUTPUT PATH based on
#   the target architecture.
X_OUTARCH := $(X_OUTTOP)/$(X_ARCH)
#
##################################################

endif
################################################################################
################################################################################



# Strip out the build system engine files from MAKEFILE_LIST
X_MAKEFILES := $(filter-out %.mak,$(MAKEFILE_LIST))

# Obtain just the module (sub-directory) name of the calling Makefile
X_MODULE := $(patsubst %/,%,$(dir $(word $(words $(X_MAKEFILES)),$(X_MAKEFILES))))


# Define OUTPUT PATH for this specific module
#   (It's object files will go here.)
$(X_MODULE)_OUTPUT := $(X_OUTARCH)/$(X_MODULE)

# Make sure that the OUTPUT PATH for the module's
#   object files exists... we do this by making
#   a call to mkdir
X_IGNORE := $(shell mkdir -p $($(X_MODULE)_OUTPUT))

# Lets create bin sub-directories while we are at it...
X_IGNORE := $(shell mkdir -p $($(X_MODULE)_OUTPUT)/bin)


################################################################################
################################################################################
ifndef X_ALREADYLOADED
X_ALREADYLOADED = 1
.PHONY: clean
clean::
	@rm -rf $(X_OUTARCH)
.SUFFIXES:
endif
################################################################################
################################################################################

