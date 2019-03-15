Name:           plastimatch
Version:        1.7.4
Release:        1%{?dist}
Summary:        Medical image registration and reconstruction

License:        BSD-style
URL:            http://plastimatch.org
Source0:        https://downloads.sourceforge.net/%{name}/%{name}-%{version}.tar.bz2

BuildRequires:  dcmtk-devel,InsightToolkit-devel,fftw
Requires:       dcmtk-devel,InsightToolkit-devel,fftw

%description


%prep
%autosetup


%build
mkdir %{_target_platform}
pushd %{_target_platform}
%cmake -DPLM_CONFIG_DEBIAN_BUILD:BOOL=ON ..
popd
%make_build -C %{_target_platform}


%install
%make_install -C %{_target_platform}

%check
pushd %{_target_platform}
ctest -V %{?_smp_mflags}
popd

%files
%license LICENSE.TXT
%doc README.TXT
doc/man/plastimatch.1
%{_bindir}/plastimatch

%changelog
* Thu Jan 10 2019 Gregory C. Sharp <gregsharp.geo@yahoo.com>
- Initial package version
