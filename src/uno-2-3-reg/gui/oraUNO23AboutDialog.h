//
#ifndef ORAUNO23ABOUTDIALOG_H
#define ORAUNO23ABOUTDIALOG_H

#include <QtGui/QDialog>
#include "ui_oraUNO23AboutDialog.h"

class QTimer;

namespace ora
{

class UNO23AboutDialog : public QDialog
{
Q_OBJECT

/*
TRANSLATOR ora::UNO23AboutDialog
*/

public:
  /** Default constructor **/
  UNO23AboutDialog(QWidget *parent = 0);
  /** Destructor **/
  ~UNO23AboutDialog();

protected:
  /** Timer for controlling the hand-animation. **/
  QTimer *m_AnimationTimer;
  /** Percentage for animation [0;100] **/
  double m_CurrentPercentage;
  /** Current direction of animation **/
  int m_CurrentDirection;

  /** Initialize the GUI components (content) **/
  void Initialize();

protected slots:
  void OnAnimationTimerTimeout();

private:
    Ui::UNO23AboutDialogClass ui;
};

}

#endif // ORAUNO23ABOUTDIALOG_H
