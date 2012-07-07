/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#include "plmregister_config.h"
#include <sstream>
#include <stdlib.h>
#include <string.h>
#include "itkRegularStepGradientDescentOptimizer.h"

#include "compiler_warnings.h"
#include "itk_image_type.h"
#include "itk_optim.h"
#include "itk_registration.h"
#include "itk_registration_private.h"
#include "plmbase.h"
#include "plmregister.h"
#include "plmsys.h"
#include "plmutil.h"

/* Lots of ITK algorithms don't behave well.  They don't keep track of 
   the best result, and/or they don't evaluate the initial position. 
   We're going to keep all this state in the observer, and the 
   registration should query the observer to find the best registration. 

   RSG
   - Invokes itk::IterationEvent
   - Current position is valid at StartEvent
   - Initial transform is not evaluated
   - Final transform is not optimal

   Amoeba
   - Invokes itk::FunctionValueIterationEvent
   - Current position is not valid at StartEvent
   - Initial transform is evaluated
   - Final transform is optimal

   LBFGS
   - Invokes itk::FunctionAndGradientEvaluationIterationEvent
   - Current position is not valid at StartEvent
*/
class Optimization_observer : public itk::Command
{
public:
    typedef Optimization_observer Self;
    typedef itk::Command Superclass;
    typedef itk::SmartPointer < Self > Pointer;
    itkNewMacro(Self);

public:
    Stage_parms* m_stage;
    RegistrationType::Pointer m_registration;
    double m_best_value;
    double m_prev_value;
    int m_feval;
    Plm_timer* timer;
    Xform m_best_xform;

protected:
    Optimization_observer() {
        m_stage = 0;
        m_feval = 0;
        m_best_value = DBL_MAX;
        m_prev_value = -DBL_MAX;
        timer = new Plm_timer;
        timer->start ();
    };
    ~Optimization_observer() {
        delete timer;
    }

public:
    void Set_Stage_parms (RegistrationType::Pointer registration,
        Stage_parms* stage)
    {
        m_registration = registration;
        m_stage = stage;
    }

    void Execute (itk::Object * caller, const itk::EventObject & event)
    {
        Execute((const itk::Object *) caller, event);
    }

    void
    Execute (const itk::Object * object, const itk::EventObject & event)
    {
        if (typeid(event) == typeid(itk::StartEvent)) {
            m_feval = 0;
            m_prev_value = -DBL_MAX;
            lprintf ("StartEvent: ");
            if (m_stage->xform_type != STAGE_TRANSFORM_BSPLINE) {
                std::stringstream ss;
                ss << optimizer_get_current_position (m_registration, m_stage);
                lprintf (ss.str().c_str());
            }
            lprintf ("\n");
            timer->start ();
        }
        else if (typeid(event) == typeid(itk::InitializeEvent)) {
            lprintf ("InitializeEvent: \n");
            timer->start ();
        }
        else if (typeid(event) == typeid(itk::EndEvent)) {
            lprintf ("EndEvent: ");
            if (m_stage->xform_type != STAGE_TRANSFORM_BSPLINE) {
                std::stringstream ss;
                ss << optimizer_get_current_position (m_registration, m_stage);
                lprintf (ss.str().c_str());
            }
            lprintf ("\n");
            lprintf ("%s\n", m_registration->GetOptimizer()
                ->GetStopConditionDescription().c_str());
        }
        else if (typeid(event) 
            == typeid(itk::FunctionAndGradientEvaluationIterationEvent))
        {
            int it = optimizer_get_current_iteration(m_registration, m_stage);
            double val = optimizer_get_value(m_registration, m_stage);
            double duration = timer->report ();

            lprintf ("%s [%2d,%3d] %9.3f [%6.3f secs]\n", 
                (m_stage->metric_type == METRIC_MSE) ? "MSE" : "MI",
                it, m_feval, val, duration);
            timer->start ();
            m_feval++;
        }
        else if (typeid(event) == typeid(itk::FunctionEvaluationIterationEvent))
        {
            lprintf ("itk::FunctionEvaluationIterationEvent\n");
        }
        else if (typeid(event) == typeid(itk::IterationEvent))
        {
            int it = optimizer_get_current_iteration(m_registration, m_stage);
            double val = optimizer_get_value(m_registration, m_stage);
            double ss = optimizer_get_step_length(m_registration, m_stage);

            lprintf (" VAL %9.3f SS %5.2f ", val, ss);

            if (m_stage->xform_type != STAGE_TRANSFORM_BSPLINE) {
                std::stringstream ss;
                ss << optimizer_get_current_position (m_registration, m_stage);
                lprintf (ss.str().c_str());
            }

            if (m_prev_value != -DBL_MAX) {
                double diff = fabs(m_prev_value - val);
                lprintf (" %10.2f", val - m_prev_value);
                if (it >= m_stage->min_its && diff < m_stage->convergence_tol) {
                    lprintf (" (tol)", val - m_prev_value);

                    /* calling StopOptimization() doesn't always stop 
                       optimization */
                    /* calling optimizer_set_max_iterations () doesn't 
                       seem to always stop rsg. */
                    if (m_stage->optim_type == OPTIMIZATION_RSG) {
                        typedef itk::RegularStepGradientDescentOptimizer 
                            * OptimizerPointer;
                        OptimizerPointer optimizer = dynamic_cast< 
                            OptimizerPointer >(m_registration->GetOptimizer());
                        optimizer->StopOptimization();
                    } else {
                        optimizer_set_max_iterations (m_registration, 
                            m_stage, 1);
                    }
                }
            }
            m_prev_value = val;

            if (val < m_best_value) {
                m_best_value = val;
                lprintf (" *");
            }

            lprintf ("\n");
        }
        else if (typeid(event) == typeid(itk::ProgressEvent)) {
            lprintf ("ProgressEvent: ");
            if (m_stage->xform_type != STAGE_TRANSFORM_BSPLINE) {
                std::stringstream ss;
                ss << optimizer_get_current_position (m_registration, m_stage);
                lprintf (ss.str().c_str());
            }
            lprintf ("\n");
        }
        else {
            lprintf ("Unknown event type: %s\n", event.GetEventName());
        }
    }
};

void
Itk_registration_private::set_observer ()
{
    typedef Optimization_observer OOType;
    OOType::Pointer observer = OOType::New();
    observer->Set_Stage_parms (registration, stage);
    registration->GetOptimizer()->AddObserver(itk::StartEvent(), observer);
    registration->GetOptimizer()->AddObserver(itk::InitializeEvent(), observer);
    registration->GetOptimizer()->AddObserver(itk::IterationEvent(), observer);
    registration->GetOptimizer()->AddObserver(itk::FunctionEvaluationIterationEvent(), observer);
    registration->GetOptimizer()->AddObserver(itk::ProgressEvent(), observer);
    registration->GetOptimizer()->AddObserver(itk::EndEvent(), observer);
}
