DICOM tutorial
==============
So, you want to learn about dicom, eh?

The only good reference I could find on the web is the DICOM standard.  
You need this, so download it now.

http://medical.nema.org/

DICOM file format
-----------------

Preamble and header
~~~~~~~~~~~~~~~~~~~
To start with, DICOM files might have a header, or they might not.  
The standard says they need a header, but older dicom files won't have this.  
If it has a header, it is in the following format:

- 128 byte file preamble, containing application-specific data
- 4 bytes containing the string "DICM"

The dicom preamble and header are described in section 10 of the standard.

Data elements
~~~~~~~~~~~~~
A dicom data set (such as a file) includes a sequence of data elements.
A single data element is composed of the following fields:

- 4 byte tag (includes 2 byte group number + 2 byte element number)
- 2 byte value representation (optional, with optional 2 byte reserved field)
- 2 or 4 byte value length
- variable length value field (with even bytes)

The layout of data elements is described in section 5 of the standard.

Value representations
~~~~~~~~~~~~~~~~~~~~~
A value representation (VR) is the "type" of the data.  For example, 
it denotes whether an item is a string, integer, UID, or sequence.

The list of value representations are described in section 5 of the standard.

Value length
~~~~~~~~~~~~
The value length is either a 2 or 4 byte unsigned integer.  
It is 2 bytes if there is an explicit VR, otherwise it is 4 bytes.

Explicit and implicit VR
~~~~~~~~~~~~~~~~~~~~~~~~
VR can be either explicit or implicit.  Explicit means that the VR is 
specified together with each object, whereas implicit means that 
the application has to lookup the correct VR from a dictionary.  
The dictionary maps (group,element) pairs to VR values.

The application describes whether it will use implicit or explicit VR 
in the application header.  A special case is made of the sequence tags
(fffe,e000), (fffe,e00d), and (fffe,e0dd) which are always implicit VR.
