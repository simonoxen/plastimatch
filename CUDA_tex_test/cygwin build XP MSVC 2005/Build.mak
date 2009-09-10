##########################################
# Note: GNU Make 3.80 or higher required #
##########################################


# Get the file type extensions for the target platform
include $(X_ARCH).mak

# Define OBJECT(S) LIST (includes module dependencies)
$(X_MODULE)_OBJS := $(addsuffix $(X_OBJEXT),$(addprefix $($(X_MODULE)_OUTPUT)/,$(basename $(SRCS)))) $(DEPS)

# Define OUTPUT BINARY
$(X_MODULE)_BINARY := $(addprefix $($(X_MODULE)_OUTPUT)/bin/,$(BINARY))$(BINARY_EXT)

# Include platform specific build rules
include $(X_ARCH)-rules.mak

all:: $($(X_MODULE)_BINARY)
$(X_MODULE): $($(X_MODULE)_BINARY)
