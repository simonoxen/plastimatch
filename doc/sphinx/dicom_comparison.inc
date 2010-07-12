DICOM Anonymizer comparison
---------------------------
There are a number of good and not-so-good dicom libraries and toolkits 
available.  However, it is not so easy to choose between them.  
For example, most DICOM anonymizers have bugs, which cause them to be 
unusable in some fashion or another.

Over the years, I've used a lot of DICOM anonymizers.  The following 
is a list of common problems that I have encountered:

- Various date fields cannot be reset
- It is not possible to preserve relative dates (for example, we would 
  like to preserve the duration between StudyDate and StructureSetDate)
- Private tags are always deleted
- Comments, diagnonsis descriptions, or other fields cannot be reset
- It is not possible to modify strings (for example, the physician 
  might type "smith-final" into the StructureSetName field, and we 
  would like to delete "smith" but preserve "final" )
- UIDs are not changed, or relationships between UIDs are not preserved

Conquest DICOM
~~~~~~~~~~~~~~
  http://www.xs4all.nl/~ingenium/dicom.html

  License: BSD-style

  Version: 1.4.15

CTN
~~~

DVTk
~~~~

GDCM v2
~~~~~~~

LONI Inspector
~~~~~~~~~~~~~~

MIRC DicomEditor
~~~~~~~~~~~~~~~~
  http://mircwiki.rsna.org/index.php?title=DicomEditor

  License: MIRC license

  Version: 22

  A pretty nice tool for batch editing of dicom files.   You can modify 
  each tag using separate rules, and all fields are accessible.

  Private fields can be preserved.

  Dates are properly anonymized.  There is a feature which allows date 
  offsets, but it only partially works (@incrementdate works fine, 
  but I could not get @offsetdate to work).

  It includes a regular expression substitution tool for strings.

  It includes a UID remapper, based on UID hashing.  However, UIDs within 
  Sequences don't seem to work (they are not remapped).

Tudor DICOM
~~~~~~~~~~~


Other information
~~~~~~~~~~~~~~~~~
The following link includes a comparison of different DICOM 
toolkits in terms of their APIs.

http://www.vph-noe.eu/vph-repository/doc_download/141-dicom-survey
