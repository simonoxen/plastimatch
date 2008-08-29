##################################################################
##  Header data
##################################################################

$head_103_part1 = <<EODATA
(0002,0010) UI =LittleEndianExplicit                    #  20, 1 TransferSyntaxUID
(0008,0005) CS [ISO_IR 100]                             #  10, 1 SpecificCharacterSet
(0008,0012) DA [%s]                               #   8, 1 InstanceCreationDate
(0008,0013) TM [%s]                                 #   6, 1 InstanceCreationTime
(0008,0014) UI [%s]                   #  20, 1 InstanceCreatorUID
(0008,0016) UI =RTStructureSetStorage                   #  30, 1 SOPClassUID
(0008,0018) UI [%s] #  52, 1 SOPInstanceUID
(0008,0020) DA [20000101]                               #   8, 1 StudyDate
(0008,0021) DA [20000101]                               #   8, 1 SeriesDate
(0008,0022) DA [20000101]                               #   8, 1 AcquisitionDate
(0008,0023) DA [20000101]                               #   8, 1 ContentDate
(0008,0030) TM [143425]                                 #   6, 1 StudyTime
(0008,0050) SH (no value available)                     #   0, 0 AccessionNumber
(0008,0060) CS [RTSTRUCT]                               #   8, 1 Modality
(0008,0070) LO [MGH]                     #  15, 1 Manufacturer
(0008,0090) PN []                                       #   2, 1 ReferringPhysiciansName
(0008,0092) ST (no value available)                     #   0, 0 ReferringPhysiciansAddress
(0008,0094) SH (no value available)                     #   0, 0 ReferringPhysiciansTelephoneNumbers
(0008,1010) SH [%s]                               #  StationName
(0008,103e) LO [Auto Structure Set]              #  SeriesDescription
(0008,1090) LO [Plastimatch]                          #  ManufacturersModelName
(0010,0010) PN [%s]                              #  10, 1 PatientsName
(0010,0020) LO [%s]                                   #   4, 1 PatientID
(0010,0030) DA [20000101]                               #   8, 1 PatientsBirthDate
(0010,0040) CS [O]                                      #   2, 1 PatientsSex
(0010,0050) SQ (Sequence with explicit length #=0)      #   0, 1 PatientsInsurancePlanCodeSequence
(fffe,e0dd) na (SequenceDelimitationItem for re-encod.) #   0, 0 SequenceDelimitationItem
(0010,1000) LO (no value available)                     #   0, 0 OtherPatientIDs
(0010,1001) PN []                                       #   2, 1 OtherPatientNames
(0010,1005) PN []                                       #   2, 1 PatientsBirthName
(0010,1010) AS [000Y]                                   #   4, 1 PatientsAge
(0010,1040) LO (no value available)                     #   0, 0 PatientsAddress
(0010,1060) PN []                                       #   2, 1 PatientsMothersBirthName
(0010,1090) LO (no value available)                     #   0, 0 MedicalRecordLocator
(0010,2154) SH (no value available)                     #   0, 0 PatientsTelephoneNumbers
(0018,0000) UL 32                                       #   4, 1 AcquisitionGroupLength
(0018,1000) LO [6100c33e]                               #   8, 1 DeviceSerialNumber
(0018,1020) LO [6.0.102]                                #   8, 1 SoftwareVersions
(0020,0000) UL 160                                      #   4, 1 ImageGroupLength
(0020,000d) UI [%s] #  52, 1 StudyInstanceUID
(0020,000e) UI [%s] #  56, 1 SeriesInstanceUID
(0020,0010) SH [%s]                                   #   4, 1 StudyID
(0020,0011) IS [%s]                                    #   4, 1 SeriesNumber
(0020,0013) IS [%s]                                    #   4, 1 InstanceNumber
EODATA
  ;

$head_103_part2 = <<EODATA
(3006,0002) SH [%s]                        #  16, 1 StructureSetLabel
(3006,0004) LO [%s]                  #  22, 1 StructureSetName
(3006,0008) DA [%s]                               #   8, 1 StructureSetDate
(3006,0009) TM [%s]                                 #   6, 1 StructureSetTime
(3006,0010) SQ (Sequence with undefined length #=1)     # u/l, 1 ReferencedFrameOfReferenceSequence
  (fffe,e000) na (Item with undefined length #=2)         # u/l, 1 Item
    (0020,0052) UI [%s] #  62, 1 FrameOfReferenceUID
    (3006,0012) SQ (Sequence with undefined length #=1)     # u/l, 1 RTReferencedStudySequence
      (fffe,e000) na (Item with undefined length #=3)         # u/l, 1 Item
        (0008,1150) UI =DetachedStudyManagementSOPClass         #  24, 1 ReferencedSOPClassUID
        (0008,1155) UI [%s] #  50, 1 ReferencedSOPInstanceUID
        (3006,0014) SQ (Sequence with undefined length #=1)     # u/l, 1 RTReferencedSeriesSequence
          (fffe,e000) na (Item with undefined length #=2)         # u/l, 1 Item
            (0020,000e) UI [%s] #  50, 1 SeriesInstanceUID
            (3006,0016) SQ (Sequence with undefined length #=144)   # u/l, 1 ContourImageSequence
EODATA
  ;

$item_103_part2 = <<EODATA
              (fffe,e000) na (Item with undefined length #=2)          # 106, 1 Item
                (0008,1150) UI =CTImageStorage                          #  26, 1 ReferencedSOPClassUID
                (0008,1155) UI [%s] #  52, 1 ReferencedSOPInstanceUID
              (fffe,e00d) na (ItemDelimitationItem)   #   0, 0 ItemDelimitationItem
EODATA
  ;

$foot_103_part2 = <<EODATA
            (fffe,e0dd) na (SequenceDelimitationItem)               #   0, 0 SequenceDelimitationItem
          (fffe,e00d) na (ItemDelimitationItem)                   #   0, 0 ItemDelimitationItem
        (fffe,e0dd) na (SequenceDelimitationItem)               #   0, 0 SequenceDelimitationItem
      (fffe,e00d) na (ItemDelimitationItem)                   #   0, 0 ItemDelimitationItem
    (fffe,e0dd) na (SequenceDelimitationItem)               #   0, 0 SequenceDelimitationItem
  (fffe,e00d) na (ItemDelimitationItem)                   #   0, 0 ItemDelimitationItem
(fffe,e0dd) na (SequenceDelimitationItem)               #   0, 0 SequenceDelimitationItem
EODATA
  ;

$head_103_part3 = <<EODATA
(3006,0020) SQ (Sequence with undefined length #=1)
EODATA
  ;

$item_103_part3 = <<EODATA
  (fffe,e000) na (Item with undefined length #=6)         # u/l, 1 Item
    (3006,0022) IS [%d]                                     #   2, 1 ROINumber
    (3006,0024) UI [%s] #  62, 1 ReferencedFrameOfReferenceUID
    (3006,0026) LO [%s]                                #   8, 1 ROIName
    (3006,0036) CS (no value available)                     #   0, 0 ROIGenerationAlgorithm
  (fffe,e00d) na (ItemDelimitationItem)                   #   0, 0 ItemDelimitationItem
EODATA
  ;

$foot_103_part3 = <<EODATA
(fffe,e0dd) na (SequenceDelimitationItem)               #   0, 0 SequenceDelimitationItem
EODATA
  ;

$head_103_part4 = <<EODATA
(3006,0039) SQ (Sequence with undefined length #=1)
EODATA
  ;

$subhead_103_part4 = <<EODATA
  (fffe,e000) na (Item with undefined length #=4)
    (3006,002a) IS [%s]         # ROIDisplayColor
    (3006,0084) IS [%s]         # ReferencedROINumber
    (3006,0040) SQ (Sequence undefined length #=108)
EODATA
  ;

$item_103_part4_with_ac = <<EODATA
      (fffe,e000) na (Item with undefined length #=7)
        (3006,0016) SQ (Sequence explicit length #=1)
          (fffe,e000) na (Item with explicit length #=3)
            (0008,1150) UI =CTImageStorage
            (0008,1155) UI [%s] #  52, 1 ReferencedSOPInstanceUID
          (fffe,e00d) na (ItemDelimitationItem for re-encoding)
        (fffe,e0dd) na (SequenceDelimitationItem for re-encod.)
        (3006,0042) CS [CLOSED_PLANAR]
        (3006,0044) DS [%s]                                    #   4, 1 ContourSlabThickness
        (3006,0046) IS [%s]                                     #   2, 1 NumberOfContourPoints
        (3006,0048) IS [%s]                                      #   2, 1 ContourNumber
        (3006,0049) IS [%s]                                      #   2, 1 AttachedContours
        (3006,0050) DS [%s] # 288,30 ContourData
      (fffe,e00d) na (ItemDelimitationItem for re-encoding)
EODATA
  ;

$item_103_part4_without_ac = <<EODATA
      (fffe,e000) na (Item with undefined length #=7)
        (3006,0016) SQ (Sequence explicit length #=1)
          (fffe,e000) na (Item with explicit length #=3)
            (0008,1150) UI =CTImageStorage
            (0008,1155) UI [%s] #  52, 1 ReferencedSOPInstanceUID
          (fffe,e00d) na (ItemDelimitationItem for re-encoding)
        (fffe,e0dd) na (SequenceDelimitationItem for re-encod.)
        (3006,0042) CS [CLOSED_PLANAR]
        (3006,0046) IS [%s]                                     #   2, 1 NumberOfContourPoints
        (3006,0048) IS [%s]                                      #   2, 1 ContourNumber
        (3006,0050) DS [%s] # 288,30 ContourData
      (fffe,e00d) na (ItemDelimitationItem for re-encoding)
EODATA
  ;

$subfoot_103_part4 = <<EODATA
    (fffe,e0dd) na (SequenceDelimitationItem for re-encod.) #   0, 0 SequenceDelimitationItem
    (3006,0084) IS [%s]                                      #   2, 1 ReferencedROINumber
  (fffe,e00d) na (ItemDelimitationItem for re-encoding)
EODATA
  ;

$foot_103_part4 = <<EODATA
(fffe,e0dd) na (SequenceDelimitationItem for re-encod.)
EODATA
  ;

$head_103_part5 = <<EODATA
(3006,0080) SQ (Sequence with undefined length #=1)
EODATA
  ;

$item_103_part5 = <<EODATA
  (fffe,e000) na (Item with undefined length #=6)
    (3006,0082) IS [%d]                                      #   2, 1 ObservationNumber
    (3006,0084) IS [%d]                                      #   2, 1 ReferencedROINumber
    (3006,0085) SH [%s]                                #   8, 1 ROIObservationLabel
    (3006,00a4) CS (no value available)                     #   0, 0 RTROIInterpretedType
    (3006,00a6) PN (no value available)                     #   0, 0 ROIInterpreter
  (fffe,e00d) na (ItemDelimitationItem for re-encoding)   #   0, 0 ItemDelimitationItem
EODATA
  ;

$foot_103_part5 = <<EODATA
(fffe,e0dd) na (SequenceDelimitationItem for re-encod.) #   0, 0 SequenceDelimitationItem
EODATA
  ;

