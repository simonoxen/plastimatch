//
#ifndef UNO23REGINFO_H_
#define UNO23REGINFO_H_

#include <string>
#include <vector>

// ORAIFTools
#include <oraStringTools.h>

namespace ora
{

/** \class UNO23REGGlobalInformation
 * \brief Global UNO-2-3-REG version and product information.
 *
 * Global UNO-2-3-REG version and product information.
 *
 * In order to retrieve global information related to UNO-2-3-REG application.
 *
 * If this product is released, the information of this class must be updated
 * (the static const members in this class' definition in *.cxx file)!!
 *
 * @author phil 
 * @version 1.0
 */
class UNO23REGGlobalInformation
{
public:
  /** Pure (numeric) product version (major.minor.release.build) **/
  static const std::string ProductVersion;
  /** Natural (string-based) product version name ("some-string"). This is meant
   * to be a major-release-specific string. **/
  static const std::string NaturalProductVersion;
  /** Copyright note. **/
  static const std::string Copyright;
  /** Company short name. **/
  static const std::string CompanyShortName;
  /** Company long name. **/
  static const std::string CompanyLongName;
  /** Application short name. **/
  static const std::string ApplicationShortName;
  /** Application long name. **/
  static const std::string ApplicationLongName;

protected:
  /** List of authors and their contributions and mail-addresses. The authors
   * are separated by '|'s and the tuple-items (name, contribution, mail) are
   * separated by '*'s. Contribution and mail are optional.
   * @see GetAuthorsInformation() **/
  static const std::string Authors;

public:

  /** Combines the pure and natural name into a single string. **/
  static std::string GetCombinedVersionString();

  /** Combines long and short company names. **/
  static std::string GetCombinedCompanyName();

  /** Combines long and short application names. **/
  static std::string GetCombinedApplicationName();

  /** Original application binary name (platform-specific). **/
  static std::string GetOriginalApplicationBinaryName();

  /** Simple structure representing author-information. **/
  typedef struct
  {
    std::string Name;
    std::string Contribution;
    std::string Mail;
  } AuthorInformation;

  /** Generate and return the array of authors (developers). **/
  static std::vector<AuthorInformation> GetAuthorsInformation();

};

}


#endif /* UNO23REGINFO_H_ */
