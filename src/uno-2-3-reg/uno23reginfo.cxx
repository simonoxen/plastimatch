//
#include "uno23reginfo.h"

namespace ora
{

// static member initialization:
const std::string UNO23REGGlobalInformation::ProductVersion = "1.2.5.1";
const std::string UNO23REGGlobalInformation::NaturalProductVersion = "3-Fingers";
const std::string UNO23REGGlobalInformation::Copyright = "Copyright (c) 2010-2011 radART";
const std::string UNO23REGGlobalInformation::CompanyShortName = "radART";
const std::string UNO23REGGlobalInformation::CompanyLongName = "Institute for Research and Development on Advanced Radiation Technologies";
const std::string UNO23REGGlobalInformation::ApplicationShortName = "UNO-2-3-REG";
const std::string UNO23REGGlobalInformation::ApplicationLongName = "Universal Open N-way 2D/3D Registration";
const std::string UNO23REGGlobalInformation::Authors = "Philipp Steininger*Registration framework, GUI, scripting, ORA-integration*-|" \
  "Markus Neuner*Mask-optimization, GUI, scripting, deployment and maintenance*-|" \
  "Heinz Deutschmann*ORA-integration*-";


std::string UNO23REGGlobalInformation::GetCombinedVersionString()
{
  if (NaturalProductVersion.length() > 0)
  {
    std::string s = "'" + NaturalProductVersion;
    s += "' (" + ProductVersion;
    s += ")";
    return s;
  }
  else
  {
    return ProductVersion;
  }
}

std::string UNO23REGGlobalInformation::GetCombinedCompanyName()
{
  if (CompanyShortName.length() > 0)
  {
    if (CompanyLongName.length() > 0)
    {
      std::string s = CompanyLongName;
      s += " (" + CompanyShortName;
      s += ")";
      return s;
    }
    else
    {
      return CompanyShortName;
    }
  }
  else
  {
    return CompanyLongName;
  }
}

std::string UNO23REGGlobalInformation::GetCombinedApplicationName()
{
  if (ApplicationShortName.length() > 0)
  {
    if (ApplicationLongName.length() > 0)
    {
      std::string s = ApplicationLongName;
      s += " (" + ApplicationShortName;
      s += ")";
      return s;
    }
    else
    {
      return ApplicationShortName;
    }
  }
  else
  {
    return ApplicationLongName;
  }
}

std::string UNO23REGGlobalInformation::GetOriginalApplicationBinaryName()
{
#if ( defined( _WIN32 ) || defined ( _WIN64 ) ) && !defined( __CYGWIN__ )
  return "uno23reg.exe";
#else
  return "uno23reg";
#endif
}

std::vector<UNO23REGGlobalInformation::AuthorInformation> UNO23REGGlobalInformation::GetAuthorsInformation()
{
  std::vector<AuthorInformation> authors;
  std::vector<std::string> authorstrings;
  std::vector<std::string> authordetails;
  AuthorInformation ai;
  Tokenize(Authors, authorstrings, "|");
  for (std::size_t i = 0; i < authorstrings.size(); i++)
  {
    authordetails.clear();
    Tokenize(authorstrings[i], authordetails, "*");
    if (authordetails.size() == 3)
    {
      ai.Name = TrimF(authordetails[0]);
      ai.Contribution = TrimF(authordetails[1]);
      ai.Mail = TrimF(authordetails[2]);
      authors.push_back(ai);
    }
  }
  return authors;
}

}
