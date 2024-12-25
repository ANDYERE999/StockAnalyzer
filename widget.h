
#define WIDGET_H

#include <QWidget>
#include <QProcess>
#include <QLineSeries>
#include <QValueAxis>
#include <QProgressdialog>

class QStandardItemModel;

QT_BEGIN_NAMESPACE
namespace Ui {
class Widget;
}
QT_END_NAMESPACE

class Widget : public QWidget
{
    Q_OBJECT

public:
    Widget(QWidget *parent = nullptr);
    ~Widget();
    void readDataFromFile(const QString &filename);
    void saveDataToFile(const QString &filename);
	void deleteFile(const QString& filename);
	void generateChart(const QString& filename, int p, int q,int m);
	void generateChartPredict(const QString& filename, int p, int q, int m);
    double calculateSharpeRatioFromFile(const QString& filename, double riskFreeRate, const QDate& startDate, const QDate& endDate);
    double calculateSharpeRatioFromFile1(const QString& filename, double riskFreeRate, const QDate& startDate, const QDate& endDate);
    void handleUpdateDatabaseError();
    void handleUpdateDatabaseOutput();
    void handleUpdateDatabaseFinished(int exitCode, QProcess::ExitStatus exitStatus);
	void fixJsonFile(const QString& filename);
    void copyTxtToCsv(const QString& sourcefile, const QString& targetfile);
private slots:
    void on_pushButton_clicked();
    void on_pushButton_3_clicked();
    void on_pushButton_2_clicked();
    void on_pushButton_4_clicked();
    void handleProcessOutput();
	void handleProcessError();
	void on_pushButtonShowText_clicked();
	void on_pushButton_5_clicked();//pushButton_5表示获取K线数据
	void handleProcessFinished(int exitCode, QProcess::ExitStatus exitStatus);
    void on_findbasicinfo_clicked();
    void on_on_pushButtonGenerateChart_clicked();

    void on_getKlineByName_clicked();

    void on_pushButton_Setting_clicked();
    void on_updateDatabase_clicked();
    

    void on_calculateSharpeRatioButton_clicked();

    void on_pushButton_predictByLine_clicked();
    void predictByLine_Kline();
	void predictByLine_KlinePlus();
    void on_pushButton_predictByAI_clicked();
    
private:
    Ui::Widget *ui;
    QStandardItemModel *model;
    QProcess *process;
    QProgressDialog* progressDialog;
};

