################################################
# CONTAINS BUILD RULES FOR: CYNWIN_NT-5.1_i686 #
#     (WindowsXP Pro - Version 2002 - SP2)     #
################################################

# Available Symbols:
#   $(X_ARCH)			Build Environment Architecture
#   $(X_OUT_ARCH)		Build Output Directory
#   $(X_MODULE)			Name of Current Module
#   $($(X_MODULE)_OUTPUT)	Module (including path to destination)
#   $($(X_MODULE)_OBJS)		Object File List (including path(s))
#   $($(X_MODULE)_BINARY)	Module Binary (including destinatino path)
#

# C Rule (.c -> .obj):
#######################################################
## OLD WAY (BROKEN) ###################################
#######################################################
#$($(X_MODULE)_OUTPUT)/%.obj: $(X_MODULE)/%.c
#	"$(CL)" /c $(CFLAGS) /I "$(VC_INC)" /Fo$@ $<
#######################################################
$($(X_MODULE)_OUTPUT)/%.obj: $(X_MODULE)/%.c
	"$(NVCC)" -I"$(CUDA_INC)" -Xcompiler "$(CFLAGS_CUDA)" $(NVCCFLAGS) -DWIN32 -D_CONSOLE -o $@ -c $<

	
# C++ Rules:
#######################################################
## OLD WAY (BROKEN) ###################################
#######################################################
# (.cpp -> .obj)
#$($(X_MODULE)_OUTPUT)/%.obj: $(X_MODULE)/%.cpp
#	"$(CL)" /c $(CFLAGS) /I "$(VC_INC)" /Fo$@ $<
#######################################################
$($(X_MODULE)_OUTPUT)/%.obj: $(X_MODULE)/%.cpp
	"$(NVCC)" -I"$(CUDA_INC)" -Xcompiler "$(CFLAGS_CUDA)" $(NVCCFLAGS) -DWIN32 -D_CONSOLE -o $@ -c $<

# (.cxx -> .obj)
#######################################################
## OLD WAY (BROKEN) ###################################
#######################################################
#$($(X_MODULE)_OUTPUT)/%.obj: $(X_MODULE)/%.cxx
#	"$(CL)" /c $(CFLAGS) /I "$(VC_INC)" /Fo$@ $<
#######################################################
$($(X_MODULE)_OUTPUT)/%.obj: $(X_MODULE)/%.cxx
	"$(NVCC)" -I"$(CUDA_INC)" -Xcompiler "$(CFLAGS_CUDA)" $(NVCCFLAGS) -DWIN32 -D_CONSOLE -o $@ -c $<


# CUDA Rule (.cu -> .cpp -> .obj):
#######################################################
# (* cl.exe must be in the environment's PATH)
# (** nvcc takes .cu -> .cpp & then cl takes .cpp -> .obj)
#######################################################
$($(X_MODULE)_OUTPUT)/%.obj: $(X_MODULE)/%.cu
	"$(NVCC)" -I"$(CUDA_INC)" -Xcompiler "$(CFLAGS_CUDA)" $(NVCCFLAGS) -DWIN32 -D_CONSOLE -o $@ -c $<
#######################################################


# Linker Rule:
#######################################################
## OLD WAY (BROKEN) ###################################
#######################################################
#$($(X_MODULE)_OUTPUT)/$(BINARY)$(X_EXEEXT): $($(X_MODULE)_OBJS)
#	"$(LINK)" $^ \
#		/LIBPATH:"$(STD_LIB_DOS)" /LIBPATH:"$(STD_LIB_CUDA_1_DOS)" /LIBPATH:"$(STD_LIB_CUDA_2_DOS)" \
#		$(STD_LIBS_CUDA) $(STD_LIBS) \
#		/OUT:"$($(X_MODULE)_OUTPUT)/bin/$(BINARY)$(X_EXEEXT)"
#######################################################
$($(X_MODULE)_OUTPUT)/bin/$(BINARY)$(X_EXEEXT): $($(X_MODULE)_OBJS)
	"$(NVCC)" $(NVCCFLAGS) $^ \
		-L "$(STD_LIB_CUDA_1)" -L "$(STD_LIB_CUDA_2)" \
		-o $@
