
#ifndef ORAITKVTKIMAGE_H_
#define ORAITKVTKIMAGE_H_

#include "oraITKVTKImageMetaInformation.h"

// ORAIFTools
#include "SimpleDebugger.h"

#include <itkImage.h>
#include <itkImageIOBase.h>
#include <itkDataObject.h>
#include <vtkImageData.h>


namespace ora
{


/**
 * This is a wrapper abstracting from concrete image representation. It is
 * capable of converting between ITK and VTK image formats. The image data is
 * always represented in 3 dimensions.
 * The ITK component type (actual type of the pixel type) template is avoided
 * by up-casting of the itk::Image smartpointer to its template free
 * itk::DataObject baseclass.
 *
 * In addition, this wrapper holds an optional pointer to a meta-information
 * structure which contains general meta-information regarding the image.
 *
 * Objects of this class can either be manually generated or automatically
 * be generated by using the static loader methods of this class.
 *
 * @see ora::ITKVTKImageMetaInformation
 *
 * @author phil 
 * @author Markus 
 * @version 1.5.1
 */
class ITKVTKImage
{
public:
  /** Important image and image-related typedefs **/
  /** Dimension of the ITK-VTK image.
   * Required for templated itk::Image at compile time.
   * NOTE: May change in the future to a non static member (e.g. 2D, 4D) and
   * can be read at runtime with the same mechanics as the component type.
   */
  static const unsigned int Dimensions = 3;
  /** Template free base-class of itk::Image to avoid the need for a class
   * template.
   */
  typedef itk::DataObject ITKImageType;
  /**
   * Template free base-class smartpointer. The itk::Image smartpointer is
   * obtained by casting the raw pointer to an itk::Image<T,Dim>::Pointer
   * smartpointer.
   */
  typedef ITKImageType::Pointer ITKImagePointer;
  typedef vtkImageData VTKImageType;
  typedef VTKImageType* VTKImagePointer;
  typedef itk::ImageIOBase::IOPixelType ITKPixelType;
  typedef itk::ImageIOBase::IOComponentType ITKComponentType;
  /** Enums used to indicate which method should be used to load an image.
   * @see LoadImageFromFile()
   */
  typedef enum {
    /** Reads an image from a file with ITK.
     * @see LoadFromOrdinaryImageUsingITK()
     */
    ITKIO,
    /** Reads an image from a file with VTK.
     * @see LoadFromOrdinaryImageUsingVTK()
     */
    VTKIO,
    /** Reads an image from an ora-xml file.
     * @see LoadFromORAMetaImage()
     */
    ORAIO} IOType;

  /** Global image file extension for open radART extended meta image format
   */
  static const std::string ORA_EXTENDED_META_IMAGE_EXTENSION;

  /** Determines the ITK \a pixelType (e.g. SCALAR, RGB, RGBA, OFFSET, VECTOR)
   * and ITK \a componentType (e.g. UCHAR, CHAR, USHORT) at runtime by using an
   * ITK dummy reader.
   * @see #m_PixelType
   * @see #m_ComponentType
   * @param fileName [in] Image filename to read type information.
   * @param pixelType [out] The ITK pixel type of the image. Is set to
   *    itk::ImageIOBase::UNKNOWNPIXELTYPE if type information cannot be read.
   * @param componentType [out] The ITK component type of the image. Is set to
   *    itk::ImageIOBase::UNKNOWNCOMPONENTTYPE if type information cannot be read.
   */
  static void GetImageTypes(const std::string fileName,
      ITKPixelType &pixelType, ITKComponentType &componentType);

  /** Converts the template type to an itk::ImageIOBase::IOComponentType.
   * @return Component type itk::ImageIOBase::IOComponentType of the template.
   *    Returns itk::ImageIOBase::UNKNOWNCOMPONENTTYPE if no type matches.
   */
  template <typename TComponentType>
  static itk::ImageIOBase::IOComponentType GetComponentTypeFromTemplate();

  /** Reads an image depending on the set \a loadType. The image type information
   * is obtained at runtime (ITKVTKimage::ITKIO, ITKVTKimage::VTKIO). Then the
   * corresponding method for reading is called with the determined
   * component type.
   * @see #IOType
   * @see GetImageTypes()
   * @see LoadFromOrdinaryImageUsingITK()
   * @see LoadFromOrdinaryImageUsingVTK()
   * @see LoadFromORAMetaImage()
   * @param fileName Filename of the image if \a loadType is ITKVTKimage::ITKIO
   *    and ITKVTKimage::VTKIO. If \a loadType is ITKVTKimage::ORAIO the xml-file
   *    must be provided.
   * @param loadType Indicates which method should be used to read the file.
   * @return Generated image data object if successful (NULL in case of
   * failure)
   */
  static ITKVTKImage *LoadImageFromFile(const std::string fileName,
      IOType loadType);

  /**
   * Generates a new ITKVTKImage from ordinary ITK-compatible image data
   * represented as file. The related meta-info object will contain minimal
   * meta-data. Generic ITK-reader technology will be used internally.
   * <br><b>NOTE:</b> There is no connection between the meta-information object
   * and the according patient (image list, FOR collection) - this information
   * must externally be set by using the appropriate methods of the
   * meta-information object.
   * This function is templated over the component type of the image.
   * @param fileName the name of ITK-compatible image data
   * @param pt ITK pixel type @see #ITKPixelType
   * @param ct ITK component type @see #ITKPixelType
   * @return Generated image data object if successful (NULL in case of
   * failure)
   */
  template <typename TComponentType>
  static ITKVTKImage *LoadFromOrdinaryImageUsingITK(
      const std::string fileName, const ITKPixelType &pt, const ITKComponentType &ct);

  /**
   * Generates a new ITKVTKImage from ordinary VTK-compatible image data
   * represented as file. The related meta-info object will contain minimal
   * meta-data. Generic VTK-reader technology will be used internally.
   * <br><b>NOTE:</b> There is no connection between the meta-information object
   * and the according patient (image list, FOR collection) - this information
   * must externally be set by using the appropriate methods of the
   * meta-information object.
   * This function is templated over the component type of the image.
   * @param fileName the name of VTK-compatible image data
   * @param pt ITK pixel type @see #ITKPixelType
   * @param ct ITK component type @see #ITKPixelType
   * @return Generated image data object if successful (NULL in case of
   * failure)
   */
  template <typename TComponentType>
  static ITKVTKImage *LoadFromOrdinaryImageUsingVTK(
      const std::string fileName, const ITKPixelType &pt, const ITKComponentType &ct);

  /**
   * Generates a new ITKVTKImage from ORA extended meta image data
   * represented as files. The related meta-info object will contain the
   * meta-data stored in the ORA meta-XML file.
   * <br><b>NOTE:</b> There is no connection between the meta-information object
   * and the according patient (image list, FOR collection) - this information
   * must externally be set by using the appropriate methods of the
   * meta-information object.
   * <br><b>NOTE:</b> The MHD file must be in the same directory as the ORA.XML
   * file. An absolute path in the ORA-XML meta-information is ignored.
   * @param fileName the name of ORA extended image data (the XML-file, not
   *    the mhd-file!)
   * @param ignoreHash if TRUE, the hash (security) of the XML-file is ignored;
   *    THIS IS NOT RECOMMENDED!
   * @param forceOrientationFromMetaInfo some previous versions stored the
   * image orientation incorrectly in the MHD file, but correctly in the XML
   * file - in order to cope with such image data, this flag should be set to
   * TRUE; if a discrepancy is detected, the MHD file is overridden!
   * @return Generated image data object if successful (NULL in case of
   *    failure)
   */
  static ITKVTKImage *LoadFromORAMetaImage(
      const std::string fileName, bool ignoreHash = false,
      bool forceOrientationFromMetaInfo = true);

  /** Constructs a new image object with ITK type information.
   *
   * @param pt ITK pixel type @see #ITKPixelType
   * @param ct ITK component type @see #ITKPixelType
   */
  explicit ITKVTKImage(ITKPixelType pt, ITKComponentType ct);

  /** Default Destructor **/
  virtual ~ITKVTKImage();

  /** @return Internal loaded image in ITK (base-class) image representation **/
  template <typename TComponentType>
   ITKImagePointer GetAsITKImage();

  /** @return Internal loaded image in VTK image representation **/
  template <typename TComponentType>
  VTKImagePointer GetAsVTKImage();

  /**
   * Set the internal reference image (ITK base-class representation).
   * Note that setting the ITK image forces resetting the current VTK image.
   * Note that the set ITK image will be updated and disconnected from its original
   * source (e.g. a reader).
   * <br><b>NOTE:</b> The image type information may lead to unexpected results
   * if it does not match the itk-vtk-image type.
   * @param itkImage pointer to the ITK base-class image which should be the
   *    source image
   * @param updateVTKImageImmediately if this parameter is TRUE, the VTK image
   *    representation is immediately updated which may save computation time
   *    later
   */
  virtual void SetITKImage(ITKImagePointer itkImage,
      bool updateVTKImageImmediately);

  /**
   * Set the internal reference image (VTK representation). Note that setting
   * the ITK image forces resetting the current ITK image. Note that the set
   * VTK image will be updated and disconnected from its original source (e.g.
   * a reader) - this is achieved by making a shallow copy of the original
   * image.
   * <br><b>NOTE:</b> The image type information may lead to unexpected results
   * if it does not match the itk-vtk-image type.
   * @param vtkImage pointer to the VTK image which should be the source image
   * @param updateITKImageImmediately if this parameter is TRUE, the ITK image
   * representation is immediately updated which may save computation time
   * later
   */
  virtual void SetVTKImage(VTKImagePointer vtkImage,
      bool updateITKImageImmediately);

  /** Extract the basic image geometry from the internal ITK image using CPP
   * data types instead of ITK-related types.
   * @param spacing returned pixel spacing
   * @param size returned image dimension
   * @param origin returned image origin
   * @param orientation returned image orientation (row-, column-, slicing-
   * directions; 9 components)
   * @return true if the returned by-ref parameters are valid **/
  template <typename TComponentType>
   bool ExtractBasicImageGeometry(double spacing[3], int size[3],
       double origin[3], double orientation[9]);

  /** Get a pointer to the internal image meta-information (open radART). **/
  virtual ITKVTKImageMetaInformation::Pointer GetMetaInfo()
  {
    return m_MetaInfo;
  }
  /** Set the internal image meta-information (open radART). **/
  virtual void SetMetaInfo(ITKVTKImageMetaInformation::Pointer mi)
  {
    m_MetaInfo = mi;
  }

  /**
   * Saves this ITKVTKImage as a physical file in ORA extended metaimage
   * format (meta-image + meta-XML).
   * @param fileName the name of destination image file (*.mhd-file); extension
   * .mhd will automatically be added if not already
   * @param compress Z-compression will be applied if TRUE (*.zraw-file)
   * @param intermediateDirectory if this parameter is not empty the image is
   * first stored (streamed) to the specified directory, then copied to the
   * real destination, then the image is deleted from the intermediate
   * directory; this can be useful when the image is stored to network resources
   * as this can be buggy during streaming; this parameter MUST HAVE A
   * TRAILING SEPARATOR!
   * This function is templated over the component type of the image.
   * @return TRUE if all ORA extended metaimage files could be successfully
   * written.
   */
  template <typename TComponentType>
  bool SaveImageAsORAMetaImage(std::string fileName,
      bool compress, std::string intermediateDirectory = "");

  /** Get the ITK pixel type (e.g. SCALAR, RGB, RGBA, OFFSET, VECTOR) of this
   * image object.
   * @see itk::ImageIOBase::IOPixelType
   */
  SimpleGetter(PixelType, ITKPixelType)
  /** Get the ITK component type (e.g. UCHAR, CHAR, USHORT) of this image object
   * that refers to the actual storage class associated with either a SCALAR
   * pixel type or elements of a compound pixel.
   * @see itk::ImageIOBase::IOComponentType
   */
  SimpleGetter(ComponentType, ITKComponentType)

protected:

  /** hosted ITK image **/
  ITKImagePointer m_ITKImage;
  /** hosted VTK image **/
  VTKImagePointer m_VTKImage;
  /** hosted image meta-information (open radART) **/
  ITKVTKImageMetaInformation::Pointer m_MetaInfo;
  /** ITK pixel type (e.g. SCALAR, RGB, RGBA, OFFSET, VECTOR)
   * @see itk::ImageIOBase::IOPixelType
   */
  ITKPixelType m_PixelType;
  /** ITK component type (e.g. UCHAR, CHAR, USHORT) that refers to the actual
   * storage class associated with either a SCALAR pixel type or elements of a
   * compound pixel.
   * @see itk::ImageIOBase::IOComponentType
   */
  ITKComponentType m_ComponentType;

  /**
   * Helper method for extracting minimal meta-information out of an
   * ordinary ITK-image-object.
   * @param i pointer to the source ITK-image
   * @param mi pointer to a valid meta-information-object that should
   * receive the infomation
   */
  template <typename TComponentType>
  static void ExtractMinimalMetaInfoFromITKImage(ITKImagePointer &i,
    ITKVTKImageMetaInformation::Pointer mi);

private:
  /** Default Constructor is private because pixel and component type are
   * required for object creation.
   */
  ITKVTKImage(){}

};


}

#include "oraITKVTKImage.txx"

#endif /* ORAITKVTKIMAGE_H_ */
