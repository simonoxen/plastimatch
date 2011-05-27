//
#ifndef ORAUNO23SPARSEREGISTRATIONEXECUTIONTASK_H_
#define ORAUNO23SPARSEREGISTRATIONEXECUTIONTASK_H_

#include "oraUNO23RegistrationExecutionTask.h"

#include <itkObject.h>
#include <itkEventObject.h>

namespace ora
{

// forward declaration:
class UNO23Model;

/**
 * // FIXME:
 *
 * @see UNO23RegistrationExecutionTask
 *
 * @author phil 
 * @version 1.0
 */
class UNO23SparseRegistrationExecutionTask : public UNO23RegistrationExecutionTask
{
public:
  /** Default constructor **/
  UNO23SparseRegistrationExecutionTask();
  /** Destructor **/
  virtual ~UNO23SparseRegistrationExecutionTask();

 /** Executes registration.
   * @return TRUE if the execution succeeded
   * @see Task#Execute()
   */
  virtual bool Execute();

  /** Unexecutes the task. Apply the previous transform.
   * @see Task#Unexecute()
   */
  virtual bool Unexecute();

  /**
   * The name of the task is "Registration".
   * @see Task#GetName()
   */
  virtual QString GetName();

  /** The number of iterations (only available AFTER Execute()!) **/
  virtual int GetNumberOfIterations()
  {
    return m_NumberOfIterations;
  }

protected:
  /** The number of iterations (only available AFTER Execute()!) **/
  int m_NumberOfIterations;
  /** Helper for ITK-based observer progress **/
  double m_CurrentProgressStart;
  /** Helper for ITK-based observer progress **/
  double m_CurrentProgressSpan;
  /** Helper for ITK-based observer progress (TRUE=EXECUTE, FALSE=UNEXECUTE) **/
  bool m_CurrentProgressDirection;

  /** Entry-point for internal ITK filter observer. **/
  void OnInternalFilterProgressReceptor(itk::Object *caller,
      const itk::EventObject &event);

};


}


#endif /* ORAUNO23SPARSEREGISTRATIONEXECUTIONTASK_H_ */
