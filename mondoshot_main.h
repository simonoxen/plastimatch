/* -----------------------------------------------------------------------
   See COPYRIGHT.TXT and LICENSE.TXT for copyright and license information
   ----------------------------------------------------------------------- */
#ifndef __mondoshot_main_h__
#define __mondoshot_main_h__

#include <wx/wx.h>
#include <wx/listctrl.h>

enum
{
    ID_MENU_QUIT = 1,
    ID_MENU_ABOUT,
    ID_BUTTON_OK,
    ID_BUTTON_CANCEL,
    ID_LISTCTRL_PATIENTS,
    ID_TEXTCTRL_PATIENT_NAME,
    ID_TEXTCTRL_PATIENT_ID
};

class MyApp: public wxApp
{
public:
    virtual bool OnInit();
};

class MyListCtrl: public wxListCtrl
{
public:
    MyListCtrl(wxWindow *parent,
               const wxWindowID id,
               const wxPoint& pos,
               const wxSize& size,
               long style)
        : wxListCtrl(parent, id, pos, size, style),
          m_attr(*wxBLUE, *wxLIGHT_GREY, wxNullFont)
        {
#ifdef __POCKETPC__
            EnableContextMenu();
#endif
        }

    // add one item to the listctrl in report mode
    void InsertItemInReportView(int i);

    void OnColClick(wxListEvent& event);
    void OnColRightClick(wxListEvent& event);
    void OnColBeginDrag(wxListEvent& event);
    void OnColDragging(wxListEvent& event);
    void OnColEndDrag(wxListEvent& event);
    void OnBeginDrag(wxListEvent& event);
    void OnBeginRDrag(wxListEvent& event);
    void OnBeginLabelEdit(wxListEvent& event);
    void OnEndLabelEdit(wxListEvent& event);
    void OnDeleteItem(wxListEvent& event);
    void OnDeleteAllItems(wxListEvent& event);
#if WXWIN_COMPATIBILITY_2_4
    void OnGetInfo(wxListEvent& event);
    void OnSetInfo(wxListEvent& event);
#endif
    void OnSelected(wxListEvent& event);
    void OnDeselected(wxListEvent& event);
    void OnListKeyDown(wxListEvent& event);
    void OnActivated(wxListEvent& event);
    void OnFocused(wxListEvent& event);
    void OnCacheHint(wxListEvent& event);

    void OnChar(wxKeyEvent& event);

#if USE_CONTEXT_MENU
    void OnContextMenu(wxContextMenuEvent& event);
#endif

    void OnRightClick(wxMouseEvent& event);

private:
    void ShowContextMenu(const wxPoint& pos);
    wxLog *m_logOld;
    void SetColumnImage(int col, int image);

    void LogEvent(const wxListEvent& event, const wxChar *eventName);
    void LogColEvent(const wxListEvent& event, const wxChar *eventName);

#if defined (commentout)
    virtual wxString OnGetItemText(long item, long column) const;
    virtual int OnGetItemColumnImage(long item, long column) const;
    virtual wxListItemAttr *OnGetItemAttr(long item) const;
#endif

    wxListItemAttr m_attr;

    DECLARE_NO_COPY_CLASS(MyListCtrl)
    DECLARE_EVENT_TABLE()
};

class MyFrame: public wxFrame
{
public:

    MyFrame(const wxString& title, const wxPoint& pos, const wxSize& size);

    virtual bool OnInit ();
    void OnMenuQuit (wxCommandEvent& event);
    void OnMenuAbout (wxCommandEvent& event);
    void OnButtonOK (wxCommandEvent& event);
    void OnButtonCancel (wxCommandEvent& event);
    void OnHotKey1 (wxKeyEvent& event);
    void OnHotKey2 (wxKeyEvent& event);
    void listctrl_patients_populate (void);

    wxBitmap m_bitmap;
    wxTextCtrl *m_textctrl_patient_name;
    wxTextCtrl *m_textctrl_patient_id;
    MyListCtrl *m_listctrl_patients;
    wxPanel *m_panel;

    DECLARE_EVENT_TABLE()
};

#endif /* __mondoshot_main_h__ */
