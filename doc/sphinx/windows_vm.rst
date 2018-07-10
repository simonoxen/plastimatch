.. _windows_vm:

Windows Virtual Machine
=======================
This section describes the method for setting up the windows
virtual machine for development.

#. Make a copy of VM clone
#. Increase memory to 8 GiB and increase to 2 CPUs
#. Log in; increase display size
#. Run windows update and reboot
#. Install windows bash
   #. Settings -> Update & Security -> For developers -> Developer Mode
   #. Control Panel -> Programs -> Turn Windows Features On and Off 
      -> Windows Subsystem for Linux
   #. Reboot
   #. Visit https://aka.ms/wslstore to install
   #. Edit /etc/passwd, change home directory to /mnt/c/Users/IEUser
   #. Update packages with sudo apt-get update && sudo apt-get dist-upgrade

#. Connect shared folder (VBox shared folder method)
   #. On windows::

	net use x: \\sherbert\vb-share /user:gcs6

   #. On linux subsystem::

	sudo mount -t drvfs x: /mnt/x

   #. But there is a permission denied error
      #. https://github.com/Microsoft/WSL/issues/2988

#. Connect shared folder (VBox shared folder method)
   #. On linux, create shared folder::

	Folder Path: /PHShome/gcs6/shared/vb-share
	Folder Name: vb-share
	Make Permanent

   #. On windows::

	net use x: \\vboxsvr\vb-share

   #. However, existing files do not have read permission
      #. https://github.com/Microsoft/WSL/issues/2896

#. When installing visual studio, you need to choose "custom" install, 
   because "recommended" install does not include C++.
#. Because there is no way to directly access the shared folders,
   they will need to be copied in Windows instead of WSL.
#. Several of the "exe" programs cannot be run from the shared drive.
   You need to copy  them onto the local drive (Desktop) before running.
#. Enable .net 3.5.1 in control panel (needed by WiX)
   #. Control Panel -> Programs -> Turn Windows Features On and Off 
      -> .NET Framework 3.5 (includes...)

