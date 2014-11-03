/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef _Rt_beam_h_
#define _Rt_beam_h_

#include "plmdose_config.h"
#include <string>

#include "aperture.h"
#include "rpl_volume.h"
#include "Rt_sobp.h"

class Rt_beam_private;
class Rt_sobp;

/*! \brief 
 * The Rt_beam class encapsulates a single SOBP Rt beam, including 
 * its associated aperture and range compensator.
 */
class PLMDOSE_API Rt_beam {
public:
    Rt_beam ();
    ~Rt_beam ();
public:
    Rt_beam_private *d_ptr;

public:
    /*! \name Inputs */
    ///@{
    /*! \brief Load PDD from XiO or txt file */
    bool load (const char* fn);

    /*! \brief Get the position of the beam source in world coordinates. */
    const double* get_source_position ();
    /*! \brief Get the x, y, or z coordinate of the beam source 
      in world coordinates. */
    double get_source_position (int dim);
    /*! \brief Set the position of the beam source in world coordinates. */
    void set_source_position (const float position[3]);
    /*! \brief Set the position of the beam source in world coordinates. */
    void set_source_position (const double position[3]);

    /*! \brief Get the position of the beam isocenter in world coordinates. */
    const double* get_isocenter_position ();
    /*! \brief Get the x, y, or z coordinate of the beam source 
      in world coordinates. */
    double get_isocenter_position (int dim);
    /*! \brief Set the position of the beam isocenter in world coordinates. */
    void set_isocenter_position (const float position[3]);
    /*! \brief Set the position of the beam isocenter in world coordinates. */
    void set_isocenter_position (const double position[3]);

    /*! \brief Add an SOBP pristine peak to this beam */
    void add_peak (
        double E0,                 /* initial ion energy (MeV) */
        double spread,             /* beam energy sigma (MeV) */
        double dres,               /* spatial resolution of bragg curve (mm)*/
        double dmax,               /* maximum w.e.d. (mm) */
        double weight);

    /*! \brief Get "detail" parameter of dose calculation algorithm */
    int get_detail () const;
    /*! \brief Set "detail" parameter of dose calculation algorithm */
    void set_detail (int detail);
    /*! \brief Get "flavor" parameter of dose calculation algorithm */
    char get_flavor () const;
    /*! \brief Set "flavor" parameter of dose calculation algorithm */
    void set_flavor (char flavor);

    /*! \brief Get "homo_approx" parameter of dose calculation algorithm */
    char get_homo_approx () const;
    /*! \brief Set "homo_approx" parameter of dose calculation algorithm */
    void set_homo_approx (char homo_approx);

    /*! \brief Get maximum depth (in mm) in SOBP curve */
    double get_sobp_maximum_depth ();

	/*! \brief Get Sobp */
	Rt_sobp::Pointer get_sobp();

	/*! \brief Get "beamWeight" parameter of dose calculation algorithm */
    float get_beamWeight () const;
    /*! \brief Set "beamWeight" parameter of dose calculation algorithm */
    void set_beamWeight (float beamWeight);

	/*! \Get proximal, distal margins and prescription */
	float get_proximal_margin();
	float get_distal_margin();
	float get_prescription_min();
	float get_prescription_max();


    /*! \brief Set/Get proximal margin; this is subtracted from the 
      minimum depth */
    void set_proximal_margin (float proximal_margin);
    /*! \brief Set distal margin; this is added onto the prescription
      maximum depth */
    void set_distal_margin (float distal_margin);
    /*! \brief Set SOBP range and modulation for prescription 
      as minimum and maximum depth (in mm) */
    void set_sobp_prescription_min_max (float d_min, float d_max);

	/* Set source size in mm */
	void set_source_size(float source_size);

	/* Get source size in mm */
	float get_source_size();

    /*! \brief Request debugging information to be written to directory */
    void set_debug (const std::string& dir);

    ///@}

    /*! \name Execution */
    ///@{
    void optimize_sobp ();          /* automatically create, weigh peaks */
    bool generate ();               /* use manually weighted peaks */
    ///@}

    /*! \name Outputs */
    ///@{
    void dump (const char* dir);     /* Print debugging information */
    float lookup_sobp_dose (float depth);
    ///@}

	/* This computes the aperture and range compensator */
	void compute_beam_modifiers ();

	/* This modifies the rpl_volume to account for aperture and range compensator */
    void apply_beam_modifiers ();

	/* Get aperture */
	Aperture::Pointer& get_aperture ();
    const Aperture::Pointer& get_aperture () const;

	/* Set/ Get target */
	Plm_image::Pointer& get_target ();
    const Plm_image::Pointer& get_target () const;
	void set_target(Plm_image::Pointer& target);

	/* Set/ Get dose_volume*/
	Plm_image::Pointer& get_dose ();
    const Plm_image::Pointer& get_dose () const;
	void set_dose(Plm_image::Pointer& dose);

	/* Set smearing */
	void set_smearing (float smearing);
	float get_smearing();

	/* Set/Get step_length */
    void set_step_length (double ray_step);
    double get_step_length();

	void set_beam_depth (float z_min, float z_max, float z_step);

	/* set the type of particle (proton, helium ions, carbon ions...)*/
    void set_particle_type(Particle_type particle_type);
	Particle_type get_particle_type();

	/* Set/Get intput file names */
	void set_aperture_in(std::string str);
	std::string get_aperture_in();

	void set_range_compensator_in(std::string str);
	std::string get_range_compensator_in();

	/* Set/Get output file names */
	void set_aperture_out(std::string str);
	std::string get_aperture_out();

	void set_proj_dose_out(std::string str);
	std::string get_proj_dose_out();

	void set_proj_img_out(std::string str);
	std::string get_proj_img_out();

	void set_range_compensator_out(std::string str);
	std::string get_range_compensator_out();

	void set_sigma_out(std::string str);
	std::string get_sigma_out();

	void set_wed_out(std::string str);
	std::string get_wed_out();

	void set_photon_energy(float energy);
	float get_photon_energy();

	void set_have_prescription(bool have_prescription);
	bool get_have_prescription();

	void set_have_manual_peaks(bool have_manual_peaks);
	bool get_have_manual_peaks();

	void copy_sobp(Rt_sobp::Pointer sobp);

public: 

	/* Volumes useful for dose calculation */
	/* raw volume */
	Rpl_volume* rpl_vol; // contains the radiologic path length along a ray
    Rpl_volume* ct_vol_density; // contains the ct_density along the ray
    Rpl_volume* sigma_vol;  // contains the sigma (lateral spread of the pencil beam - used to calculate the off-axis term) along the ray
    	
    /* larger volumes for Hong and divergent geometry algorithms */
    Rpl_volume* rpl_vol_lg;
    Rpl_volume* ct_vol_density_lg;
    Rpl_volume* sigma_vol_lg;
	Rpl_volume* rpl_dose_vol; // contains the dose vol for the divergent geometry algorithm

	/* aperture 3D volume to avoid artefacts*/
	Rpl_volume* aperture_vol;

private:
    bool load_xio (const char* fn);
    bool load_txt (const char* fn);

};

#endif
