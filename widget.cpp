#include "widget.h"
#include "ui_widget.h"
#include <QFile>
#include <QDebug>
#include <QTextStream>
#include <QStandardItemModel>
#include <QProcess>
#include <QInputDialog>
#include <QMessageBox>
#include <QJsonObject>
#include <QJsonDocument>
#include <QJsonArray>
#include <QDialog>
#include <QDateEdit>
#include <QLineEdit>
#include <QPushButton>
#include <QVBoxLayout>
#include <QtCharts/QChartView>
#include <QtCharts/QLineSeries>
#include <QtCharts/QScatterSeries>
#include <QtCharts/QValueAxis>
#include <QPen>
#include <QDateTime>
#include <QDateTimeAxis>
#include <limits>
#include <QFileDialog>
#include <qDesktopServices>
#include <QDir>
#include <QUrl>
#include <cmath>
#include <vector>
#include <QDialog>
#include <QLabel>
#include <QCheckBox>
#include <QVBoxLayout>
#include <QPushButton>
#include <QComboBox>
#include <QProgressDialog>
//以下是配置数据
QString databaseTime;
double riskFreeRate;
bool isVIP;
QString dataMode;
QStringList dataModeList;
QString version;
int conditionCode = 0;
bool isDebugMode = false;
void readJsonData(QString& databaseTime, double& riskFreeRate, bool& isVIP, QString& dataMode, QStringList& dataModeList, QString& version) {
    QFile file("./files/config.json");
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qDebug() << "无法打开文件进行读取";
        return;
    }

    QByteArray data = file.readAll();
    file.close();

    QJsonDocument jsonDoc = QJsonDocument::fromJson(data);
    if (jsonDoc.isNull() || !jsonDoc.isObject()) {
        qDebug() << "JSON解析失败";
        return;
    }

    // 在读取新的 dataModeList 之前清空
    dataModeList.clear();

    QJsonObject jsonObject = jsonDoc.object();
    if (jsonObject.contains("databaseTime") && jsonObject["databaseTime"].isString()) {
        databaseTime = jsonObject["databaseTime"].toString();
    }
    else {
        qDebug() << "databaseTime字段不存在或不是字符串";
    }

    if (jsonObject.contains("riskFreeRate") && jsonObject["riskFreeRate"].isDouble()) {
        riskFreeRate = jsonObject["riskFreeRate"].toDouble();
    }
    else {
        qDebug() << "riskFreeRate字段不存在或不是双精度浮点数";
    }

    if (jsonObject.contains("isVIP") && jsonObject["isVIP"].isBool()) {
        isVIP = jsonObject["isVIP"].toBool();
    }
    else {
        qDebug() << "isVIP字段不存在或不是布尔值";
    }

    if (jsonObject.contains("dataMode") && jsonObject["dataMode"].isString()) {
        dataMode = jsonObject["dataMode"].toString();
    }
    else {
        qDebug() << "dataMode字段不存在或不是字符串";
    }

    if (jsonObject.contains("dataModeList") && jsonObject["dataModeList"].isArray()) {
        QJsonArray jsonArray = jsonObject["dataModeList"].toArray();
        for (const QJsonValue& value : jsonArray) {
            if (value.isString()) {
                dataModeList.append(value.toString());
            }
        }
    }
    else {
        qDebug() << "dataModeList字段不存在或不是数组";
    }

    if (jsonObject.contains("version") && jsonObject["version"].isString()) {
        version = jsonObject["version"].toString();
    }
    else {
        qDebug() << "version字段不存在或不是字符串";
    }
}
//构造函数
Widget::Widget(QWidget* parent)//初始化Widget
    : QWidget(parent)
    , ui(new Ui::Widget)
	, process(new QProcess(this))//初始化QProcess
{
    ui->setupUi(this);
	// 读取配置数据
    readJsonData(databaseTime, riskFreeRate, isVIP, dataMode, dataModeList, version);
	//修复bug补丁:config.json被意外篡改
    fixJsonFile("./files/config.json");
    //隐藏开发者按钮
	ui->pushButton->hide();
	ui->pushButton_2->hide();
    ui->pushButton_predictByAI->hide();
    ui->pushButton_4->hide();
    // 设置支持中文的字体
    conditionCode = 0;
    setWindowIcon(QIcon(":/appicon.ico"));
    QFont font("Microsoft YaHei", 10);
    ui->tableView->setFont(font);
    readDataFromFile(".//files//informations.txt");
	ui->labelOutput->setText("欢迎使用股票数据分析系统");
}
//析构函数
Widget::~Widget()
{
    delete ui;
}
//创建一个类，用于获取用户输入的日期和文本
class DateDialog : public QDialog
{
    Q_OBJECT

public:
    explicit DateDialog(QWidget* parent = nullptr);
    QDate getStartDate() const;
    QDate getEndDate() const;
    QString getText() const;

private:
    QDateEdit* startDateEdit;
    QDateEdit* endDateEdit;
    QLineEdit* textInput;
    QPushButton* okButton;
};
DateDialog::DateDialog(QWidget* parent) : QDialog(parent)
{
    startDateEdit = new QDateEdit(this);
    endDateEdit = new QDateEdit(this);
    textInput = new QLineEdit(this);
    okButton = new QPushButton(tr("确定"), this);

    startDateEdit->setDate(QDate::currentDate());
    endDateEdit->setDate(QDate::currentDate());

    QVBoxLayout* layout = new QVBoxLayout(this);
    layout->addWidget(startDateEdit);
    layout->addWidget(endDateEdit);
    layout->addWidget(textInput);
    layout->addWidget(okButton);

    connect(okButton, &QPushButton::clicked, this, &QDialog::accept);

    setLayout(layout);
}
QDate DateDialog::getStartDate() const
{
    return startDateEdit->date();
}
QDate DateDialog::getEndDate() const
{
    return endDateEdit->date();
}
QString DateDialog::getText() const
{
    return textInput->text();
}
//定义函数：修复bug补丁：防止json文件被意外被篡改
void Widget::fixJsonFile(const QString& filepath) {
	QFile file(filepath);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qInfo() << "无法打开文件";
        return ;
    }
    QByteArray fileData = file.readAll();
    file.close();
    QJsonDocument document = QJsonDocument::fromJson(fileData);
    if (document.isNull() || !document.isObject()) {
        qWarning() << "JSON解析失败";
        return ;
    }
    // 获取根对象
    QJsonObject rootObject = document.object();

    // 修改dataModeList的值
    QJsonArray dataModeList;
    dataModeList.append("online");
    dataModeList.append("offline");
    rootObject["dataModeList"] = dataModeList;

    // 将修改后的JSON对象写回文件
    document.setObject(rootObject);
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qWarning() << "无法打开文件进行写入";
        return ;
    }
    file.write(document.toJson());
    file.close();

    qDebug() << "dataModeList已成功修改";
}
//复制txt，变为csv
void Widget::copyTxtToCsv(const QString& sourcefile, const QString& targetfile)
{
    // 定义源目录和目标目录
    QString sourceDir = sourcefile;
    QString targetDir = targetfile;

    // 确保目标目录存在
    QDir().mkpath(targetDir);

    // 获取当前时间戳
    QString timestamp = QDateTime::currentDateTime().toString("yy-MM-dd HH-mm-ss");

    // 获取源目录下所有的txt文件
    QDir dir(sourceDir);
    QStringList filters;
    filters << "*.txt";
    QFileInfoList fileList = dir.entryInfoList(filters, QDir::Files | QDir::NoSymLinks);

    // 遍历所有txt文件并复制到目标目录，重命名为指定格式
    foreach(QFileInfo fileInfo, fileList) {
        QString sourceFilePath = fileInfo.absoluteFilePath();
        QString targetFileName = QString("%1 form.csv").arg(timestamp);
        QString targetFilePath = QString("%1/%2").arg(targetDir).arg(targetFileName);

        // 复制文件并重命名
        QFile::copy(sourceFilePath, targetFilePath);
    }

    // 提示操作完成
    ui->labelOutput->setText("文件复制并重命名成功");
}
//定义计算boll线函数
void calculateBoll(const QVector<double>& closePrices, int period, QVector<double>& middleBand, QVector<double>& upperBand, QVector<double>& lowerBand, double k = 2.0) {
    int size = closePrices.size();
    middleBand.resize(size);
    upperBand.resize(size);
    lowerBand.resize(size);

    for (int i = 0;i < size;i++) {
        if (i < period - 1) {
            middleBand[i] = upperBand[i] = lowerBand[i] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        double sum = 0.0;
        for (int j = i - period + 1;j <= i;j++) {
            sum += closePrices[j];
        }
        double mean = sum / period;
        middleBand[i] = mean;

        double variance = 0.0;
        for (int j = i - period + 1; j <= i;++j) {
            variance += std::pow(closePrices[j] - mean, 2);
        }
        double stddev = std::sqrt(variance / period);
        upperBand[i] = mean + k * stddev;
        lowerBand[i] = mean - k * stddev;
    }
}
// 读取文件并计算BOLL线
void fileCalculateBollLine(const QString& filename, int a, int b, int c) {
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qDebug() << "无法打开文件进行读取";
        return;
    }

    QTextStream in(&file);
    QStringList headers = in.readLine().split(",");
    QVector<QStringList> data;
    QVector<double> closePrices;

    while (!in.atEnd()) {
        QStringList line = in.readLine().split(",");
        data.append(line);
        closePrices.append(line[6].toDouble()); // 读取收盘价
    }
    file.close();

    QVector<double> middleBandA, upperBandA, lowerBandA;
    QVector<double> middleBandB, upperBandB, lowerBandB;
    QVector<double> middleBandC, upperBandC, lowerBandC;

    calculateBoll(closePrices, a, middleBandA, upperBandA, lowerBandA);
    calculateBoll(closePrices, b, middleBandB, upperBandB, lowerBandB);
    calculateBoll(closePrices, c, middleBandC, upperBandC, lowerBandC);

    // 将计算结果添加到数据中
    for (int i = 0; i < data.size(); ++i) {
        data[i].append(QString::number(middleBandA[i], 'f', 4));
        data[i].append(QString::number(upperBandA[i], 'f', 4));
        data[i].append(QString::number(lowerBandA[i], 'f', 4));
        data[i].append(QString::number(middleBandB[i], 'f', 4));
        data[i].append(QString::number(upperBandB[i], 'f', 4));
        data[i].append(QString::number(lowerBandB[i], 'f', 4));
        data[i].append(QString::number(middleBandC[i], 'f', 4));
        data[i].append(QString::number(upperBandC[i], 'f', 4));
        data[i].append(QString::number(lowerBandC[i], 'f', 4));
    }

    // 写回文件
    if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
        qDebug() << "无法打开文件进行写入";
        return;
    }

    QTextStream out(&file);
    headers.append(QString("MiddleBand%1days").arg(a));
    headers.append(QString("UpperBand%1days").arg(a));
    headers.append(QString("LowerBand%1days").arg(a));
    headers.append(QString("MiddleBand%1days").arg(b));
    headers.append(QString("UpperBand%1days").arg(b));
    headers.append(QString("LowerBand%1days").arg(b));
    headers.append(QString("MiddleBand%1days").arg(c));
    headers.append(QString("UpperBand%1days").arg(c));
    headers.append(QString("LowerBand%1days").arg(c));
    
	
    out << headers.join(",") << "\n";

    for (const QStringList& line : data) {
        out << line.join(",") << "\n";
    }
    file.close();
}
//定义函数：读取文件
void Widget::readDataFromFile(const QString &filename){
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly)){
        qInfo()<<filename<<"Open failed";
        return;
    } else{
        qInfo()<<filename<<"Open Successfully";
    }
    QTextStream stream(&file);
    
    //创建一个模型
    model = new QStandardItemModel(this);
    //把模型交给视图
    ui->tableView->setModel(model);//setmodel所需的数据类型是

    //读取表头
    QStringList headers=stream.readLine().split(",");
    model->setHorizontalHeaderLabels(headers);

    //读取数据
    while (!stream.atEnd()){
        QStringList linedata=stream.readLine().split(",");
        QList<QStandardItem*> items;
        for (QString data : linedata)
        {
            items.push_back(new QStandardItem(data));
        }
        model->appendRow(items);
    }
}
//槽函数：按钮0被按下
void Widget::on_pushButton_clicked()
{
    qInfo()<<"添加学生-被点击了";
    //给表格视图添加空行
    model->setRowCount(model->rowCount()+1);
    //把表格视图滚动到最低下
    ui->tableView->scrollToBottom();
}

//定义函数：写入文件(目前没有被用到)
void Widget::saveDataToFile(const QString &filename){
    QFile file(filename);
    if (!file.open(QIODevice::WriteOnly)){
        qInfo()<<"saveDatataTofile failed!";
    } else{
        qInfo()<<"successfuly save student";
    }
    QTextStream stream(&file);

    //保存表头
    QString Headers;
    int HeadersAccount=model->columnCount();
    int RowAccount=model->rowCount();
    for (int i=0;i<(HeadersAccount);i++){
        //格式化成一个字符串，然后写入文件
        Headers+=(model->horizontalHeaderItem(i)->text());
        if (i!=HeadersAccount-1){Headers+=(",");}
    }
    Headers.push_back("\n");
    stream<<Headers;

    //保存数据
    for (int r=0;r<RowAccount;r++){
        QString LineData="";
        for (int i=0;i<HeadersAccount;i++){
            if (!model->item(r,1)){qInfo()<<"Data saved!But there are empty lines";goto end;}
            LineData.push_back(model->item(r,i)->text());
            if (i!=HeadersAccount-1){LineData.push_back(",");}
        }
        stream<<LineData<<"\n";

    }
    end:;
}
//定义函数：为用户打开windows文件资源管理器
void openFolderInWindows(const QString & FolderPath) {
	QDesktopServices::openUrl(QUrl::fromLocalFile(FolderPath));
}
//输出工作表到文件夹并用Windows资源管理器打开
void Widget::on_pushButton_3_clicked()
{
	if (isVIP == false) {
		QMessageBox::warning(this, tr("提示"), tr("您不是VIP用户，无法使用此功能"));
		return;
	}
	qInfo() << "保存数据-被按下";
    QString timestamp = QDateTime::currentDateTime().toString("yy-MM-dd HH-mm-ss");
	QString location1;
	QString location2;
	if (conditionCode == 1||conditionCode==3) {
		location1 = ".\\files\\QTRequireData\\kLine.txt";
		location2 = "./outputData/" + timestamp + " kLine form.csv";
	}
    else if (conditionCode == 2||conditionCode==4) {
		location1 = ".\\files\\QTRequireData\\kLine+.txt";
		location2 = "./outputData/" + timestamp + " kLine form.csv";
    }
    else if (conditionCode == -1) {
		location1 = ".\\files\\QTRequireData\\GetBasicInfo.txt";
		location2 = "./outputData/" + timestamp + " informations form.csv";
	}
    else if (conditionCode == 0) {
		QMessageBox::information(this, tr("提示"), tr("当前表格不可被保存"));
        return;
    }
    copyTxtToCsv(location1, location2);
	openFolderInWindows("./outputData");
    ui->labelOutput->setText("表格导出成功");
}
void Widget::on_pushButton_2_clicked()
{
    qInfo()<<"删除学生-被按下";
    //获取当前选中的学生
    QModelIndex index = ui->tableView->currentIndex();
    int row=index.row();
    int col=0;
    //删除它
    model->removeRow(row);

}
void Widget::on_pushButton_4_clicked()
{
    qInfo()<<"获取数据被按下";
    QString filename = "./1.py";
	QProcess process;
	process.start("python", QStringList() << filename);



}
void Widget::handleProcessFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    Q_UNUSED(exitCode);
    Q_UNUSED(exitStatus);

    QByteArray output = process->readAllStandardOutput();
    ui->labelOutput->setText(output);
}
void Widget::handleProcessOutput()
{
    QByteArray output = process->readAllStandardOutput();
    qInfo() << "Python Output:" << output;
}

void Widget::handleProcessError()
{
    QByteArray error = process->readAllStandardError();
    qInfo() << "Python Error:" << error;
}

void Widget::on_pushButtonShowText_clicked()
{
    QByteArray error = process->readAllStandardError();
	qInfo() << "显示文本被按下";
	ui->labelOutput->setText(error);
}
void Widget::deleteFile(const QString& filename) {
    QFile file(filename);
    if (file.exists()) {
        if (file.remove()) {
            qInfo() << filename << "删除成功";
        }
        else {
            qInfo() << filename << "删除失败";
        }
    }
    else {
        qInfo() << filename << "文件不存在";
    }
}

//槽函数：获取基本信息按钮被按下
void Widget::on_findbasicinfo_clicked()
{
    bool ok;
    QString text = QInputDialog::getText(this, tr("输入数据"), tr("请输入数据"), QLineEdit::Normal, "", &ok);
    if (ok && !text.isEmpty()) {
        // 用户按下确定并输入了数据
        QStringList dataList = text.split(' ', Qt::SkipEmptyParts);
        if (text == "/developermodetrue") {
			qInfo() << "进入调试模式";
			isDebugMode = true;
			ui->labelOutput->setText("进入调试模式");
			ui->pushButton->show();
			ui->pushButton_2->show();
			ui->pushButton_4->show();
			ui->pushButton_predictByAI->show();
			readJsonData(databaseTime, riskFreeRate, isVIP, dataMode, dataModeList, version);
			return;
        }
        QJsonArray jsonArray;
        for (const QString &item : dataList) {
            jsonArray.append(item);
        }
        QJsonObject jsonObject;
        jsonObject["stock_name"] = jsonArray;
        QJsonDocument jsonDoc1(jsonObject);
        QString jsonString = jsonDoc1.toJson(QJsonDocument::Compact);
        QFile file1("./files/pyData/GetBasicInfo.json");
        if (!file1.open(QIODevice::WriteOnly | QIODevice::Text)) {
            qDebug() << "无法打开文件进行写入";
        }
        file1.write(jsonDoc1.toJson());
        file1.close();

        qInfo() << "text:" << text;
		//删除原有的数据
		QString TargetDeleteFile = "./files/QTRequireData/GetBasicInfo.txt";
		deleteFile(TargetDeleteFile);


        // 打开exe文件
        process->start("./GetBasicInfo.exe");
        if (!process->waitForStarted()) {
            QMessageBox::warning(this, tr("错误"), tr("无法启动exe文件"));
        }
        else {
            qInfo() << "打开成功";
            // 等待exe文件运行结束
            if (process->waitForFinished()) {
                qInfo() << "exe文件运行结束";
            }
            else {
                QMessageBox::warning(this, tr("错误"), tr("exe文件运行过程中出错"));
            }
        }
		//打开表格并显示数据
		readDataFromFile("./files/QTRequireData/GetBasicInfo.txt");
		ui->labelOutput->setText("基本信息已成功显示");
        conditionCode = -1;
    }
    // 用户按下取消或未输入数据，对话框自动消失
}

//槽函数：获取K线数据（通过股票代码）按钮被按下
void Widget::on_pushButton_5_clicked()
{
    // 创建一个对话框
    QDialog dialog(this);
    dialog.setWindowTitle("获取K线数据");

    // 创建控件
    QLabel* labelStartDate = new QLabel("开始日期:", &dialog);
    QDateEdit* startDateEdit = new QDateEdit(QDate::currentDate().addMonths(-1), &dialog);
    startDateEdit->setCalendarPopup(true); // 启用日历弹出
    startDateEdit->setDisplayFormat("yyyy-MM-dd"); // 设置日期显示格式

    QLabel* labelEndDate = new QLabel("结束日期:", &dialog);
    QDateEdit* endDateEdit = new QDateEdit(QDate::currentDate(), &dialog);
    endDateEdit->setCalendarPopup(true);
    endDateEdit->setDisplayFormat("yyyy-MM-dd");

    QLabel* labelText = new QLabel("股票代码:", &dialog);
    QLineEdit* textInput = new QLineEdit(&dialog);
    textInput->setPlaceholderText("请输入股票代码，如 sh.600519");

    QPushButton* okButton = new QPushButton("确定", &dialog);
    QPushButton* cancelButton = new QPushButton("取消", &dialog);

    // 布局设置
    QGridLayout* layout = new QGridLayout(&dialog);
    layout->addWidget(labelStartDate, 0, 0);
    layout->addWidget(startDateEdit, 0, 1);
    layout->addWidget(labelEndDate, 1, 0);
    layout->addWidget(endDateEdit, 1, 1);
    layout->addWidget(labelText, 2, 0);
    layout->addWidget(textInput, 2, 1);
    layout->addWidget(okButton, 3, 0);
    layout->addWidget(cancelButton, 3, 1);

    // 连接信号与槽
    connect(okButton, &QPushButton::clicked, &dialog, &QDialog::accept);
    connect(cancelButton, &QPushButton::clicked, &dialog, &QDialog::reject);

    // 显示对话框并处理结果
    if (dialog.exec() == QDialog::Accepted) {
        // 获取用户输入
        QDate startDate = startDateEdit->date();
        QDate endDate = endDateEdit->date();
        QString text = textInput->text();
        QString startDateString = startDate.toString("yyyy-MM-dd");
        QString endDateString = endDate.toString("yyyy-MM-dd");
        qInfo() << "Start Date:" << startDateString;
        qInfo() << "End Date:" << endDateString;
        qInfo() << "Text:" << text;
        // 将数据写入json文件:K线数据
        ui->labelOutput->setText("获取K线数据：读入股票代码和日期");
        // 格式化意向股票代码
        QStringList text1 = text.split(' ', Qt::SkipEmptyParts);
        QJsonArray jsonArray;
        for (const QString& item : text1) {
            jsonArray.append(item);
        }
        QJsonObject jsonObject;
        jsonObject["start_date"] = startDateString;
        jsonObject["end_date"] = endDateString;
        jsonObject["fields"] = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST";
        jsonObject["stock_id"] = jsonArray;
        QJsonDocument jsonDoc1(jsonObject);
        QString jsonString = jsonDoc1.toJson(QJsonDocument::Compact);
        QFile file1("./files/pyData/kLine.json");
        if (!file1.open(QIODevice::WriteOnly | QIODevice::Text)) {
            qDebug() << "无法打开文件进行写入";
        }
        file1.write(jsonDoc1.toJson());
        file1.close();
        // 删除原有的数据
        QString TargetDeleteFile = "./files/QTRequireData/kLine.txt";
        deleteFile(TargetDeleteFile);

        // 创建进度对话框
        QProgressDialog progressDialog("加载中……", "取消", 0, 0, this);
        progressDialog.setWindowModality(Qt::WindowModal);
        progressDialog.setCancelButton(nullptr);
        progressDialog.show();

        // 打开exe文件
        ui->labelOutput->setText("获取k线数据：查询数据库");
        process->start("./kLine.exe");
        while (process->state() == QProcess::Running) {
            QCoreApplication::processEvents();
        }
        if (process->exitStatus() == QProcess::NormalExit) {
            qInfo() << "exe文件运行结束";
        }
        else {
            QMessageBox::warning(this, tr("错误"), tr("exe文件运行过程中出错"));
        }
        progressDialog.close();

        //得到股票的Boll线数据
        ui->labelOutput->setText("获取k线数据：计算Boll线");
        fileCalculateBollLine("./files/QTRequireData/kLine.txt", 5, 10, 20);
        // 打开表格并显示数据
        readDataFromFile("./files/QTRequireData/kLine.txt");
        ui->labelOutput->setText("股票K线成功获得");
        conditionCode = 1;
    }
    else {
        qInfo() << "用户取消了操作";
    }
}
//定义函数：生成折线图
void Widget::generateChart(const QString& filename, int p, int q,int m)
{
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly)) {
        qInfo() << filename << "Open failed";
        return;
    }

    QTextStream stream(&file);
    QStringList headers = stream.readLine().split(",");
    QStringList secondline = stream.readLine().split(",");

    // 修改部分：创建系列，包含收盘价和 BOLL 线
    QLineSeries* series = new QLineSeries(); // 收盘价系列
    QLineSeries* middleBandSeries = new QLineSeries(); // 中轨线系列
    QLineSeries* upperBandSeries = new QLineSeries();  // 上轨线系列
    QLineSeries* lowerBandSeries = new QLineSeries();  // 下轨线系列

    QChart* chart = new QChart();
    chart->setTitle("股票趋势折线图");
    double minY = std::numeric_limits<double>::max();
    double maxY = std::numeric_limits<double>::min();

    while (!stream.atEnd()) {
        QStringList linedata = stream.readLine().split(",");
        if (linedata.size() > q && linedata.size() > p) {
            bool okX, okY;
            QDateTime x = QDateTime::fromString(linedata[p], "yyyy-MM-dd");
            double y = linedata[q].toDouble(&okY); // 收盘价
            if (x.isValid() && okY) {
                double xValue = x.toMSecsSinceEpoch();
                series->append(xValue, y);

                // 读取 BOLL 线的数据（根据您文件中 BOLL 线的位置调整索引）
                // 假设 MiddleBand5days 在第 15 列（索引为 14），UpperBand5days 在第 16 列（索引为 15），LowerBand5days 在第 17 列（索引为 16）
                double middleBand = linedata[p + 14].toDouble();
                double upperBand = linedata[p + 15].toDouble();
                double lowerBand = linedata[p + 16].toDouble();

                middleBandSeries->append(xValue, middleBand);
                upperBandSeries->append(xValue, upperBand);
                lowerBandSeries->append(xValue, lowerBand);

                // 更新 Y 轴范围
                minY = std::min({ minY, y, middleBand, upperBand, lowerBand });
                maxY = std::max({ maxY, y, middleBand, upperBand, lowerBand });
            }
        }
    }

    // 设置收盘价系列的样式
    series->setName("收盘价");
    QPen pen(Qt::black);
    pen.setWidth(1);
    series->setPen(pen);

    // 设置 BOLL 中轨线系列的样式
    middleBandSeries->setName("BOLL 中轨线");
    QPen middlePen(Qt::blue);
    middlePen.setWidth(1);
    middleBandSeries->setPen(middlePen);

    // 设置 BOLL 上轨线系列的样式
    upperBandSeries->setName("BOLL 上轨线");
    QPen upperPen(Qt::red);
    upperPen.setWidth(1);
    upperBandSeries->setPen(upperPen);

    // 设置 BOLL 下轨线系列的样式
    lowerBandSeries->setName("BOLL 下轨线");
    QPen lowerPen(Qt::green);
    lowerPen.setWidth(1);
    lowerBandSeries->setPen(lowerPen);

    // 将系列添加到图表
    chart->addSeries(series);
    chart->addSeries(middleBandSeries);
    chart->addSeries(upperBandSeries);
    chart->addSeries(lowerBandSeries);

    chart->setTitle(secondline[m] + "趋势折线图");
    chart->createDefaultAxes();

    // 设置 X 轴
    QDateTimeAxis* axisX = new QDateTimeAxis;
    axisX->setTitleText(headers[p]);
    axisX->setFormat("yyyy-MM-dd");
    axisX->setTickCount(10); // 设置 X 轴刻度数量
    chart->setAxisX(axisX, series);
    chart->setAxisX(axisX, middleBandSeries);
    chart->setAxisX(axisX, upperBandSeries);
    chart->setAxisX(axisX, lowerBandSeries);

    // 设置 Y 轴
    QValueAxis* axisY = new QValueAxis;
    axisY->setTitleText(headers[q]);
    axisY->setRange(minY * 0.95, maxY * 1.05); // 为了美观，扩大范围
    axisY->setTickCount(11); // 设置 Y 轴刻度数量
    axisY->setLabelFormat("%.2f"); // 设置 Y 轴标签格式
    chart->setAxisY(axisY, series);
    chart->setAxisY(axisY, middleBandSeries);
    chart->setAxisY(axisY, upperBandSeries);
    chart->setAxisY(axisY, lowerBandSeries);

    QChartView* chartView = new QChartView(chart);
    chartView->setStatusTip("股票趋势折线图");
    chartView->setRenderHint(QPainter::Antialiasing);

    // 添加导出按钮（保持您的原有功能）
    QPushButton* exportButton = new QPushButton("导出为.jpg", chartView);
    connect(exportButton, &QPushButton::clicked, [chartView]() {
        QPixmap pixmap(chartView->size());
        QPainter painter(&pixmap);
        chartView->render(&painter);
        QDir().mkpath("./outputData");
        QString timestamp = QDateTime::currentDateTime().toString("yy-MM-dd HH-mm-ss");
        QString filename = QString("./outputData/%1 chart.jpg").arg(timestamp);
        pixmap.save(filename);
        QMessageBox::information(chartView, "导出成功", "折线图已成功导出为.jpg文件");
        QString folderPath = "./outputData";
        openFolderInWindows(folderPath);
        });

    QVBoxLayout* layout = new QVBoxLayout;
    layout->addWidget(chartView);
    QWidget* chartWidget = new QWidget;
    chartWidget->setLayout(layout);
    chartWidget->resize(800, 600);
    chartWidget->show();
}
void Widget::generateChartPredict(const QString& filename, int p, int q, int m)
{
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qDebug() << filename << "打开失败";
        return;
    }

    QTextStream stream(&file);
    QStringList headers = stream.readLine().split(",");
    QString codeLabel = "";

    // 获取 BOLL 线的列索引
    int MBIndex = headers.indexOf("MiddleBand5days");
    int UBIndex = headers.indexOf("UpperBand5days");
    int LBIndex = headers.indexOf("LowerBand5days");

    if (MBIndex == -1 || UBIndex == -1 || LBIndex == -1) {
        qDebug() << "BOLL 线数据缺失";
        // 视需求决定是返回还是继续
        // 这里假设继续绘制没有 BOLL 线的图表
    }

    QLineSeries* historicalSeries = new QLineSeries();
    QLineSeries* predictionSeries = new QLineSeries();

    // 创建 BOLL 线系列
    QLineSeries* middleBandSeries = new QLineSeries(); // 中轨线系列
    QLineSeries* upperBandSeries = new QLineSeries();  // 上轨线系列
    QLineSeries* lowerBandSeries = new QLineSeries();  // 下轨线系列

    QChart* chart = new QChart();

    double minY = std::numeric_limits<double>::max();
    double maxY = std::numeric_limits<double>::min();
    double minX = std::numeric_limits<double>::max();
    double maxX = std::numeric_limits<double>::min();

    // 用于记录历史数据的最后一个点
    double lastHistoricalX = 0.0;
    double lastHistoricalY = 0.0;

    while (!stream.atEnd()) {
        QStringList linedata = stream.readLine().split(",");
        if (linedata.size() > q && linedata.size() > p) {
            // 解析日期
            QString dateString = linedata[p];
            bool isPrediction = false;
            if (dateString.contains("<#Predict>")) {
                dateString = dateString.remove("<#Predict>");
                isPrediction = true;
            }
            QDateTime x = QDateTime::fromString(dateString, "yyyy-MM-dd");
            if (!x.isValid()) continue;

            // 解析 y 值（收盘价）
            bool okY;
            double y = linedata[q].toDouble(&okY);
            if (!okY) continue;

            double xValue = x.toMSecsSinceEpoch();

            if (isPrediction) {
                predictionSeries->append(xValue, y);
            }
            else {
                historicalSeries->append(xValue, y);
                // 更新最后一个历史数据点
                lastHistoricalX = xValue;
                lastHistoricalY = y;

                // 解析 BOLL 线数据
                bool okMB = false, okUB = false, okLB = false;
                double mb = 0.0, ub = 0.0, lb = 0.0;

                if (MBIndex != -1 && UBIndex != -1 && LBIndex != -1
                    && linedata.size() > LBIndex) {

                    mb = linedata[MBIndex].toDouble(&okMB);
                    ub = linedata[UBIndex].toDouble(&okUB);
                    lb = linedata[LBIndex].toDouble(&okLB);

                    if (okMB && okUB && okLB) {
                        middleBandSeries->append(xValue, mb);
                        upperBandSeries->append(xValue, ub);
                        lowerBandSeries->append(xValue, lb);

                        // 更新 Y 轴范围
                        minY = std::min({ minY, y, mb, ub, lb });
                        maxY = std::max({ maxY, y, mb, ub, lb });
                    }
                    else {
                        // 如果 BOLL 数据无效，仅考虑 y 值
                        minY = std::min(minY, y);
                        maxY = std::max(maxY, y);
                    }
                }
                else {
                    // 不存在 BOLL 数据，仅考虑 y 值
                    minY = std::min(minY, y);
                    maxY = std::max(maxY, y);
                }
            }

            minX = std::min(minX, xValue);
            maxX = std::max(maxX, xValue);

            if (codeLabel.isEmpty() && linedata.size() > m) {
                codeLabel = linedata[m]; // 获取股票代码或名称
            }
        }
    }
    file.close();

    // 设置历史数据系列的样式
    historicalSeries->setName("历史数据");
    QPen historicalPen(Qt::black);
    historicalPen.setWidth(2);
    historicalSeries->setPen(historicalPen);

    // 设置预测数据系列的样式
    predictionSeries->setName("预测值");
    QPen predictionPen(Qt::gray);
    predictionPen.setStyle(Qt::DashLine);
    predictionPen.setWidth(2);
    predictionSeries->setPen(predictionPen);

    // 在预测系列的开头插入历史数据的最后一个点
    if (!predictionSeries->points().isEmpty()) {
        predictionSeries->insert(0, QPointF(lastHistoricalX, lastHistoricalY));
    }

    // 设置 BOLL 线系列的样式
    middleBandSeries->setName("BOLL 中轨线");
    QPen middlePen(Qt::green);
    middlePen.setWidth(1);
    middleBandSeries->setPen(middlePen);

    upperBandSeries->setName("BOLL 上轨线");
    QPen upperPen(Qt::red);
    upperPen.setWidth(1);
    upperBandSeries->setPen(upperPen);

    lowerBandSeries->setName("BOLL 下轨线");
    QPen lowerPen(Qt::blue);
    lowerPen.setWidth(1);
    lowerBandSeries->setPen(lowerPen);

    // 将系列添加到图表
    chart->addSeries(historicalSeries);
    if (!predictionSeries->points().isEmpty()) {
        chart->addSeries(predictionSeries);
    }
    chart->addSeries(middleBandSeries);
    chart->addSeries(upperBandSeries);
    chart->addSeries(lowerBandSeries);

    // 设置坐标轴
    QDateTimeAxis* axisX = new QDateTimeAxis;
    axisX->setTitleText(headers[p]);
    axisX->setFormat("yyyy-MM-dd");
    axisX->setTickCount(10); // 设置 X 轴刻度数量
    axisX->setRange(QDateTime::fromMSecsSinceEpoch(minX), QDateTime::fromMSecsSinceEpoch(maxX)); // 设置 X 轴范围
    chart->addAxis(axisX, Qt::AlignBottom);
    historicalSeries->attachAxis(axisX);
    predictionSeries->attachAxis(axisX);
    middleBandSeries->attachAxis(axisX);
    upperBandSeries->attachAxis(axisX);
    lowerBandSeries->attachAxis(axisX);

    QValueAxis* axisY = new QValueAxis;
    axisY->setTitleText(headers[q]);
    axisY->setRange(minY * 0.95, maxY * 1.05); // 设置 Y 轴范围
    axisY->setTickCount(11); // 设置 Y 轴刻度数量
    axisY->setLabelFormat("%.2f"); // 设置 Y 轴标签格式
    chart->addAxis(axisY, Qt::AlignLeft);
    historicalSeries->attachAxis(axisY);
    predictionSeries->attachAxis(axisY);
    middleBandSeries->attachAxis(axisY);
    upperBandSeries->attachAxis(axisY);
    lowerBandSeries->attachAxis(axisY);

    // 设置图表标题
    chart->setTitle(QString("%1 %2趋势折线图").arg(codeLabel).arg(headers[q]));

    // 创建图表视图
    QChartView* chartView = new QChartView(chart);
    chartView->setRenderHint(QPainter::Antialiasing);

    // 添加导出按钮
    QPushButton* exportButton = new QPushButton("导出为.jpg", chartView);
    connect(exportButton, &QPushButton::clicked, [chartView]() {
        QPixmap pixmap(chartView->size());
        QPainter painter(&pixmap);
        chartView->render(&painter);
        QDir().mkpath("./outputData");
        QString timestamp = QDateTime::currentDateTime().toString("yy-MM-dd HH-mm-ss");
        QString filename = QString("./outputData/%1 chart.jpg").arg(timestamp);
        pixmap.save(filename);
        QMessageBox::information(chartView, "导出成功", "折线图已成功导出为.jpg文件");
        QString folderPath = "./outputData";
        openFolderInWindows(folderPath);
        });

    QVBoxLayout* layout = new QVBoxLayout;
    layout->addWidget(chartView);
    QWidget* chartWidget = new QWidget;
    chartWidget->setLayout(layout);
    chartWidget->resize(800, 600);
    chartWidget->show();
}
//槽函数：生成折线图按钮被按下
void Widget::on_on_pushButtonGenerateChart_clicked()
{
    
	if (conditionCode == 1) {
		generateChart("./files/QTRequireData/kLine.txt", 0, 5,1);
	}
	else if (conditionCode == 2) {
		generateChart("./files/QTRequireData/kLine+.txt", 1, 6,0);
    }
    else if (conditionCode == 3) {
		generateChartPredict("./files/QTRequireData/kLine.txt", 0, 5, 1);
	}
    else if (conditionCode == 4) {
		generateChartPredict("./files/QTRequireData/kLine+.txt", 1, 6, 0);
    }
	else if (conditionCode == 0){
		QMessageBox::warning(this, tr("错误"), tr("请先获取K线数据"));
		ui->labelOutput->setText("请先获取K线数据");
        return;
	}
}

//槽函数：获取K线数据（通过股票名称）按钮被按下
void Widget::on_getKlineByName_clicked()
{
    // 创建一个对话框
    QDialog dialog(this);
    dialog.setWindowTitle("获取K线数据");

    // 创建控件
    QLabel* labelStartDate = new QLabel("开始日期:", &dialog);
    QDateEdit* startDateEdit = new QDateEdit(QDate::currentDate().addMonths(-1), &dialog);
    startDateEdit->setCalendarPopup(true); // 启用日历弹出
    startDateEdit->setDisplayFormat("yyyy-MM-dd"); // 设置日期显示格式

    QLabel* labelEndDate = new QLabel("结束日期:", &dialog);
    QDateEdit* endDateEdit = new QDateEdit(QDate::currentDate(), &dialog);
    endDateEdit->setCalendarPopup(true);
    endDateEdit->setDisplayFormat("yyyy-MM-dd");

    QLabel* labelText = new QLabel("股票名称:", &dialog);
    QLineEdit* textInput = new QLineEdit(&dialog);
    textInput->setPlaceholderText("请输入股票名称，如 浦发银行");

    QPushButton* okButton = new QPushButton("确定", &dialog);
    QPushButton* cancelButton = new QPushButton("取消", &dialog);

    // 布局设置
    QGridLayout* layout = new QGridLayout(&dialog);
    layout->addWidget(labelStartDate, 0, 0);
    layout->addWidget(startDateEdit, 0, 1);
    layout->addWidget(labelEndDate, 1, 0);
    layout->addWidget(endDateEdit, 1, 1);
    layout->addWidget(labelText, 2, 0);
    layout->addWidget(textInput, 2, 1);
    layout->addWidget(okButton, 3, 0);
    layout->addWidget(cancelButton, 3, 1);

    // 连接信号与槽
    connect(okButton, &QPushButton::clicked, &dialog, &QDialog::accept);
    connect(cancelButton, &QPushButton::clicked, &dialog, &QDialog::reject);

    // 显示对话框并处理结果
    if (dialog.exec() == QDialog::Accepted) {
        // 获取用户输入
        QDate startDate = startDateEdit->date();
        QDate endDate = endDateEdit->date();
        QString text = textInput->text();
        QString startDateString = startDate.toString("yyyy-MM-dd");
        QString endDateString = endDate.toString("yyyy-MM-dd");
        qInfo() << "Start Date:" << startDateString;
        qInfo() << "End Date:" << endDateString;
        qInfo() << "Text:" << text;
        // 将数据写入json文件:K线数据
        ui->labelOutput->setText("获取K线数据：读入股票代码和日期");
        // 格式化意向股票代码
        QStringList text1 = text.split(' ', Qt::SkipEmptyParts);
        QJsonArray jsonArray;
        for (const QString& item : text1) {
            jsonArray.append(item);
        }
        QJsonObject jsonObject;
        jsonObject["start_date"] = startDateString;
        jsonObject["end_date"] = endDateString;
        jsonObject["fields"] = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST";
        jsonObject["stock_name"] = jsonArray;
        QJsonDocument jsonDoc1(jsonObject);
        QString jsonString = jsonDoc1.toJson(QJsonDocument::Compact);
        QFile file1("./files/pyData/getKline+.json");
        if (!file1.open(QIODevice::WriteOnly | QIODevice::Text)) {
            qDebug() << "无法打开文件进行写入";
        }
        file1.write(jsonDoc1.toJson());
        file1.close();
        // 删除原有的数据
        QString TargetDeleteFile = "./files/QTRequireData/kLine+.txt";
        deleteFile(TargetDeleteFile);

        // 创建进度对话框
        QProgressDialog progressDialog("加载中……", "取消", 0, 0, this);
        progressDialog.setWindowModality(Qt::WindowModal);
        progressDialog.setCancelButton(nullptr);
        progressDialog.show();

        // 打开exe文件
        ui->labelOutput->setText("获取k线数据：查询数据库");
        process->start("./getKline+.exe");
        while (process->state() == QProcess::Running) {
            QCoreApplication::processEvents();
        }
        if (process->exitStatus() == QProcess::NormalExit) {
            qInfo() << "exe文件运行结束";
        }
        else {
            QMessageBox::warning(this, tr("错误"), tr("exe文件运行过程中出错"));
        }
        progressDialog.close();

        //得到股票的Boll线数据
        ui->labelOutput->setText("获取k线数据：计算Boll线");
        fileCalculateBollLine("./files/QTRequireData/kLine+.txt", 5, 10, 20);
        // 打开表格并显示数据
        readDataFromFile("./files/QTRequireData/kLine+.txt");
        ui->labelOutput->setText("股票K线成功获得");
        conditionCode = 2;
    }
    else {
        qInfo() << "用户取消了操作";
    }
}
//槽函数：设置按钮被按下
void Widget::on_pushButton_Setting_clicked()
{
    qInfo() << "设置按钮被按下";
    //修复bug补丁:config.json被意外篡改
    fixJsonFile("./files/config.json");
    QDialog* dialog = new QDialog(this);
    dialog->setWindowTitle("设置");
    dialog->resize(400, 300);

    QVBoxLayout* layout = new QVBoxLayout(dialog);

    // 创建一个水平布局，将数据库时间和按钮放在一行
    QHBoxLayout* dbLayout = new QHBoxLayout();
    QLabel* label1 = new QLabel("本地数据库截至时间：", dialog);
    QLabel* labelDatabaseTime = new QLabel(databaseTime, dialog);
    QPushButton* updateButton = new QPushButton("更新本地数据库", dialog);
    updateButton->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed); // 调整按钮大小
    dbLayout->addWidget(label1);
    dbLayout->addWidget(labelDatabaseTime);
    dbLayout->addStretch(); // 添加弹性空间
    dbLayout->addWidget(updateButton);

    layout->addLayout(dbLayout);

    // 创建一个水平布局，将系统时间放在下一行
    QHBoxLayout* sysTimeLayout = new QHBoxLayout();
    QLabel* labelSysTime = new QLabel("当前系统时间：", dialog);
    QLabel* labelCurrentTime = new QLabel(QDate::currentDate().toString("yyyy-MM-dd"), dialog);
    sysTimeLayout->addWidget(labelSysTime);
    sysTimeLayout->addWidget(labelCurrentTime);
    layout->addLayout(sysTimeLayout);

    // 检查日期差值
    QDate dbDate = QDate::fromString(databaseTime, "yyyy-MM-dd");
    QDate currentDate = QDate::currentDate();
    int daysDiff = dbDate.daysTo(currentDate);
    if (daysDiff > 5) {
        QLabel* warningLabel = new QLabel("若要使用本地数据库模式，请及时更新本地数据库", dialog);
        QFont font = warningLabel->font();
        font.setPointSize(8);
        warningLabel->setFont(font);

        // 设置文字为红色
        QPalette palette = warningLabel->palette();
        palette.setColor(QPalette::WindowText, Qt::red);
        warningLabel->setPalette(palette);

        layout->addWidget(warningLabel);
    }

    // 其他设置项
    QLabel* label2 = new QLabel("默认无风险利率：", dialog);
    QLineEdit* lineEdit2 = new QLineEdit(dialog);
    lineEdit2->setText(QString::number(riskFreeRate));
    layout->addWidget(label2);
    layout->addWidget(lineEdit2);

    QLabel* label3 = new QLabel("是否VIP：", dialog);
    QCheckBox* checkBox = new QCheckBox(dialog);
    checkBox->setChecked(isVIP);
    layout->addWidget(label3);
    layout->addWidget(checkBox);

    QLabel* label4 = new QLabel("使用在线数据库/本地数据库：", dialog);
    QComboBox* comboBox = new QComboBox(dialog);
    comboBox->clear(); // 清空 QComboBox
    comboBox->addItems(dataModeList);
    comboBox->setCurrentText(dataMode);
    layout->addWidget(label4);
    layout->addWidget(comboBox);

    QPushButton* okButton = new QPushButton("确定", dialog);
    layout->addWidget(okButton);
    dialog->setLayout(layout);

    // 连接按钮的信号与槽
    connect(okButton, &QPushButton::clicked, [this, dialog, lineEdit2, checkBox, comboBox]() {
        riskFreeRate = lineEdit2->text().toDouble();
        isVIP = checkBox->isChecked();
        dataMode = comboBox->currentText();

        QJsonObject jsonObject;
        jsonObject["databaseTime"] = databaseTime;
        jsonObject["riskFreeRate"] = riskFreeRate;
        jsonObject["isVIP"] = isVIP;
        jsonObject["dataMode"] = dataMode;
        jsonObject["dataModeList"] = QJsonArray::fromStringList(dataModeList);
        jsonObject["version"] = version;
        QJsonDocument jsonDoc(jsonObject);
        QFile file("./files/config.json");
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            qDebug() << "无法打开文件进行写入";
        }
        file.write(jsonDoc.toJson());
        file.close();
        dialog->close();
        });
    readJsonData(databaseTime, riskFreeRate, isVIP, dataMode, dataModeList, version);
    fixJsonFile("./files/config.json");
    // 连接更新按钮的信号与槽
    connect(updateButton, &QPushButton::clicked, this, &Widget::on_updateDatabase_clicked);

    dialog->exec();
}
// 更新本地数据库的槽函数
void Widget::on_updateDatabase_clicked()
{
    qInfo() << "更新本地数据库按钮被按下";
    QString filePath = "./UpdatedatabaseClient.exe";
    if (!QFile::exists(filePath)) {
        QMessageBox::warning(this, tr("错误"), tr("文件 %1 不存在").arg(filePath));
        return;
    }
    openFolderInWindows(filePath);    
}
void Widget::handleUpdateDatabaseFinished(int exitCode, QProcess::ExitStatus exitStatus)
{
    Q_UNUSED(exitCode);
    Q_UNUSED(exitStatus);

    qInfo() << "更新数据库客户端运行结束";
}
void Widget::handleUpdateDatabaseOutput()
{
    QProcess* process = qobject_cast<QProcess*>(sender());
    if (process) {
        QByteArray output = process->readAllStandardOutput();
        qInfo() << "更新数据库客户端输出:" << output;
    }
}
void Widget::handleUpdateDatabaseError()
{
    QProcess* process = qobject_cast<QProcess*>(sender());
    if (process) {
        QByteArray error = process->readAllStandardError();
        qInfo() << "更新数据库客户端错误:" << error;
    }
}
//槽函数：计算夏普比率按钮被按下
void Widget::on_calculateSharpeRatioButton_clicked()
{
    if (conditionCode == 0 || conditionCode == -1) {
		QMessageBox::information(this, "提示", "请先获取K线数据");
        
    }
    else if (conditionCode == 2 || conditionCode == 4) {
        // 创建一个对话框，获取无风险利率和时间范围
        QDialog dialog(this);
        dialog.setWindowTitle("计算夏普比率");

        QLabel* labelRate = new QLabel("无风险利率 (%):", &dialog);
        QLineEdit* lineEditRate = new QLineEdit(&dialog);
        lineEditRate->setPlaceholderText("例如：3.0");

        QLabel* labelStartDate = new QLabel("开始日期:", &dialog);
        QDateEdit* startDateEdit = new QDateEdit(QDate::currentDate().addMonths(-1), &dialog);
        startDateEdit->setCalendarPopup(true);

        QLabel* labelEndDate = new QLabel("结束日期:", &dialog);
        QDateEdit* endDateEdit = new QDateEdit(QDate::currentDate(), &dialog);
        endDateEdit->setCalendarPopup(true);

        QPushButton* okButton = new QPushButton("确定", &dialog);
        QPushButton* cancelButton = new QPushButton("取消", &dialog);

        QGridLayout* layout = new QGridLayout(&dialog);
        layout->addWidget(labelRate, 0, 0);
        layout->addWidget(lineEditRate, 0, 1);
        layout->addWidget(labelStartDate, 1, 0);
        layout->addWidget(startDateEdit, 1, 1);
        layout->addWidget(labelEndDate, 2, 0);
        layout->addWidget(endDateEdit, 2, 1);
        layout->addWidget(okButton, 3, 0);
        layout->addWidget(cancelButton, 3, 1);

        connect(okButton, &QPushButton::clicked, &dialog, &QDialog::accept);
        connect(cancelButton, &QPushButton::clicked, &dialog, &QDialog::reject);

        if (dialog.exec() == QDialog::Accepted) {
            // 获取用户输入
            double riskFreeRate = lineEditRate->text().toDouble() / 100.0; // 转换为小数
            QDate startDate = startDateEdit->date();
            QDate endDate = endDateEdit->date();

            // 调用计算夏普比率的函数
            double sharpeRatio = calculateSharpeRatioFromFile1("./files/QTRequireData/kLine+.txt", riskFreeRate, startDate, endDate);

            // 显示结果
            QMessageBox::information(this, "夏普比率计算结果", QString("夏普比率为: %1").arg(sharpeRatio));
        }
    }
    else if (conditionCode == 1 || conditionCode == 3) {
        // 创建一个对话框，获取无风险利率和时间范围
        QDialog dialog(this);
        dialog.setWindowTitle("计算夏普比率");

        QLabel* labelRate = new QLabel("无风险利率 (%):", &dialog);
        QLineEdit* lineEditRate = new QLineEdit(&dialog);
        lineEditRate->setPlaceholderText("例如：3.0");

        QLabel* labelStartDate = new QLabel("开始日期:", &dialog);
        QDateEdit* startDateEdit = new QDateEdit(QDate::currentDate().addMonths(-1), &dialog);
        startDateEdit->setCalendarPopup(true);

        QLabel* labelEndDate = new QLabel("结束日期:", &dialog);
        QDateEdit* endDateEdit = new QDateEdit(QDate::currentDate(), &dialog);
        endDateEdit->setCalendarPopup(true);

        QPushButton* okButton = new QPushButton("确定", &dialog);
        QPushButton* cancelButton = new QPushButton("取消", &dialog);

        QGridLayout* layout = new QGridLayout(&dialog);
        layout->addWidget(labelRate, 0, 0);
        layout->addWidget(lineEditRate, 0, 1);
        layout->addWidget(labelStartDate, 1, 0);
        layout->addWidget(startDateEdit, 1, 1);
        layout->addWidget(labelEndDate, 2, 0);
        layout->addWidget(endDateEdit, 2, 1);
        layout->addWidget(okButton, 3, 0);
        layout->addWidget(cancelButton, 3, 1);

        connect(okButton, &QPushButton::clicked, &dialog, &QDialog::accept);
        connect(cancelButton, &QPushButton::clicked, &dialog, &QDialog::reject);

        if (dialog.exec() == QDialog::Accepted) {
            // 获取用户输入
            double riskFreeRate = lineEditRate->text().toDouble() / 100.0; // 转换为小数
            QDate startDate = startDateEdit->date();
            QDate endDate = endDateEdit->date();

            // 调用计算夏普比率的函数
            double sharpeRatio = calculateSharpeRatioFromFile1("./files/QTRequireData/kLine.txt", riskFreeRate, startDate, endDate);

            // 显示结果
            QMessageBox::information(this, "夏普比率计算结果", QString("夏普比率为: %1").arg(sharpeRatio));
        }
    }
    
}
double Widget::calculateSharpeRatioFromFile1(const QString& filename, double riskFreeRate, const QDate& startDate, const QDate& endDate)
{
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qDebug() << "无法打开文件进行读取";
        return 0.0;
    }

    QTextStream in(&file);

    // 读取表头，建立列名到索引的映射
    QString headerLine = in.readLine();
    QStringList headers = headerLine.split(",");
    QMap<QString, int> columnIndices;
    for (int i = 0; i < headers.size(); ++i) {
        columnIndices[headers[i]] = i;
    }

    // 检查必需的列是否存在
    QStringList requiredColumns = { "date", "close" };
    for (const QString& col : requiredColumns) {
        if (!columnIndices.contains(col)) {
            qDebug() << "数据缺少必要的列：" << col;
            return 0.0;
        }
    }

    int dateCol = columnIndices["date"];
    int closeCol = columnIndices["close"];

    QVector<double> returns;
    QVector<QDate> dates;
    QVector<double> closePrices;

    // 读取数据行
    while (!in.atEnd()) {
        QString line = in.readLine();
        QStringList fields = line.split(",");
        if (fields.size() != headers.size()) {
            // 数据行列数不匹配，跳过此行
            continue;
        }

        QDate date = QDate::fromString(fields[dateCol], "yyyy-MM-dd");
        if (!date.isValid()) {
            qDebug() << "无效的日期格式：" << fields[dateCol];
            continue;
        }

        if (date < startDate || date > endDate) {
            // 日期不在指定范围内，跳过
            continue;
        }

        bool ok;
        double closePrice = fields[closeCol].toDouble(&ok);
        if (!ok) {
            qDebug() << "无效的收盘价：" << fields[closeCol];
            continue;
        }

        dates.append(date);
        closePrices.append(closePrice);
    }

    file.close();

    // 检查是否有足够的数据
    if (closePrices.size() < 2) {
        qDebug() << "数据不足，无法计算收益率";
        return 0.0;
    }

    // 按日期排序（如果数据可能无序）
    QList<QPair<QDate, double>> datePricePairs;
    for (int i = 0; i < dates.size(); ++i) {
        datePricePairs.append(qMakePair(dates[i], closePrices[i]));
    }
    std::sort(datePricePairs.begin(), datePricePairs.end(), [](const QPair<QDate, double>& a, const QPair<QDate, double>& b) {
        return a.first < b.first;
        });

    // 重新整理排序后的价格列表
    closePrices.clear();
    for (const auto& pair : datePricePairs) {
        closePrices.append(pair.second);
    }

    // 计算日收益率
    for (int i = 1; i < closePrices.size(); ++i) {
        double ret = (closePrices[i] - closePrices[i - 1]) / closePrices[i - 1];
        returns.append(ret);
    }

    // 计算平均收益率和标准差
    if (returns.isEmpty()) {
        qDebug() << "没有足够的数据计算收益率";
        return 0.0;
    }

    double avgReturn = std::accumulate(returns.begin(), returns.end(), 0.0) / returns.size();
    double variance = 0.0;
    for (double ret : returns) {
        variance += (ret - avgReturn) * (ret - avgReturn);
    }
    variance /= returns.size();
    double stdDev = std::sqrt(variance);

    // 计算夏普比率
    double sharpeRatio = 0.0;
    if (stdDev != 0.0) {
        sharpeRatio = (avgReturn - riskFreeRate / 252) / stdDev; // 假设一年252个交易日
    }
    else {
        qDebug() << "标准差为0，无法计算夏普比率";
    }

    qDebug() << "夏普比率：" << sharpeRatio;
    return sharpeRatio;
}
// 槽函数：线性回归预测按钮被按下
void Widget::on_pushButton_predictByLine_clicked()
{   
	if (isVIP == false) {
		QMessageBox::warning(this, tr("错误"), tr("您不是VIP用户，无法使用此功能"));
		return;
	}
    if (conditionCode == 1) {
		predictByLine_Kline();
		conditionCode = 3;
		readDataFromFile("./files/QTRequireData/kLine.txt");
        ui->tableView->scrollToBottom();
	}
    else if (conditionCode == 2) {
		predictByLine_KlinePlus();
		conditionCode = 4;
		readDataFromFile("./files/QTRequireData/kLine+.txt");
        ui->tableView->scrollToBottom();
    }
    else if (conditionCode == 0) {
		QMessageBox::warning(this, tr("错误"), tr("请先获取K线数据"));
		return;
    }
	else if (conditionCode == 3||conditionCode==4) {
		QMessageBox::warning(this, tr("错误"), tr("已经进行过线性回归预测"));
        return;
	}
}
void Widget::predictByLine_Kline() {
    // 读取文件
    QString filename = "./files/QTRequireData/kLine.txt";
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qDebug() << "无法打开文件进行读取";
        return;
    }

    QTextStream in(&file);
    QStringList headers = in.readLine().split(",");

    // 读取第一条数据行
    QStringList firstDataLine = in.readLine().split(",");

    // 获取股票代码（假设股票代码在第二列，索引为1）
    QString code = "";
    if (firstDataLine.size() >= 2) {
        code = firstDataLine[1];
    }
    else {
        qDebug() << "数据格式错误，无法获取股票代码";
        file.close();
        return;
    }

    // 定义存储数据的容器
    QVector<QDate> dates;
    QVector<double> opens, highs, lows, closes, precloses, volumes, amounts;

    // 解析数据
    while (!in.atEnd()) {
        QStringList line = in.readLine().split(",");
        if (line.size() < 9) continue; // 确保数据完整

        QDate date = QDate::fromString(line[0], "yyyy-MM-dd");
        if (!date.isValid()) continue;

        dates.append(date);
        opens.append(line[2].toDouble());
        highs.append(line[3].toDouble());
        lows.append(line[4].toDouble());
        closes.append(line[5].toDouble());
        precloses.append(line[6].toDouble());
        volumes.append(line[7].toDouble());
        amounts.append(line[8].toDouble());
    }
    file.close();

    // 将日期转换为数值（天数）
    QVector<double> x_values;
    QDate start_date = dates.first();
    for (const QDate& date : dates) {
        x_values.append(start_date.daysTo(date));
    }

    // 对每个列进行线性回归
    auto linearRegression = [](const QVector<double>& x, const QVector<double>& y, double& a, double& b) {
        int n = x.size();
        if (n != y.size() || n == 0) {
            a = b = 0;
            return;
        }
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
        for (int i = 0; i < n; ++i) {
            sum_x += x[i];
            sum_y += y[i];
            sum_xy += x[i] * y[i];
            sum_xx += x[i] * x[i];
        }
        double denominator = n * sum_xx - sum_x * sum_x;
        if (denominator == 0) {
            a = b = 0;
            return;
        }
        a = (n * sum_xy - sum_x * sum_y) / denominator;
        b = (sum_y * sum_xx - sum_x * sum_xy) / denominator;
        };

    double a_open, b_open;
    double a_high, b_high;
    double a_low, b_low;
    double a_close, b_close;
    double a_preclose, b_preclose;
    double a_volume, b_volume;
    double a_amount, b_amount;

    linearRegression(x_values, opens, a_open, b_open);
    linearRegression(x_values, highs, a_high, b_high);
    linearRegression(x_values, lows, a_low, b_low);
    linearRegression(x_values, closes, a_close, b_close);
    linearRegression(x_values, precloses, a_preclose, b_preclose);
    linearRegression(x_values, volumes, a_volume, b_volume);
    linearRegression(x_values, amounts, a_amount, b_amount);

    // 计算后三个交易日的日期（排除周末）
    QVector<QDate> predict_dates;
    QDate last_date = dates.last();
    while (predict_dates.size() < 3) {
        last_date = last_date.addDays(1);
        if (last_date.dayOfWeek() >= 1 && last_date.dayOfWeek() <= 5) { // 周一到周五
            predict_dates.append(last_date);
        }
    }

    // 将预测值写入文件
    if (!file.open(QIODevice::Append | QIODevice::Text)) {
        qDebug() << "无法打开文件进行写入";
        return;
    }
    QTextStream out(&file);

    for (const QDate& date : predict_dates) {
        double x = start_date.daysTo(date);
        QString date_str = date.toString("yyyy-MM-dd") + "<#Predict>";
        //QString code = "sh.600007"; // 假设股票代码不变

        double open_pred = a_open * x + b_open;
        double high_pred = a_high * x + b_high;
        double low_pred = a_low * x + b_low;
        double close_pred = a_close * x + b_close;
        double preclose_pred = a_preclose * x + b_preclose;
        double volume_pred = a_volume * x + b_volume;
        double amount_pred = a_amount * x + b_amount;

        QStringList data;
        data << date_str;
        data << code;
        data << QString::number(open_pred, 'f', 4);
        data << QString::number(high_pred, 'f', 4);
        data << QString::number(low_pred, 'f', 4);
        data << QString::number(close_pred, 'f', 4);
        data << QString::number(preclose_pred, 'f', 4);
        data << QString::number(volume_pred, 'f', 0);
        data << QString::number(amount_pred, 'f', 4);

        // 补充剩余列，填充为#predict
        int total_columns = headers.size();
        for (int i = data.size(); i < total_columns; ++i) {
            data << "#predict";
        }
        out << data.join(",") << "\n";
    }
    file.close();

    // 读取更新的数据并显示
    readDataFromFile(filename);
    ui->tableView->scrollToBottom();
    ui->labelOutput->setText("线性回归预测完成！");
}
void Widget::predictByLine_KlinePlus() {
    // 读取文件
    QString filename = "./files/QTRequireData/kLine+.txt";
    QFile file(filename);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        qDebug() << "无法打开文件进行读取";
        return;
    }

    QTextStream in(&file);
    QStringList headers = in.readLine().split(",");

    // 读取第一条数据行
    QStringList firstDataLine = in.readLine().split(",");

    // 获取股票代码（假设股票代码在第三列，索引为1）
    QString code = "";
    if (firstDataLine.size() >= 3) {
        code = firstDataLine[2];
    }
    else {
        qDebug() << "数据格式错误，无法获取股票代码";
        file.close();
        return;
    }
	//获取股票名称（假设股票名称在第一列，索引为1）
    QString codeName = "";
	if (firstDataLine.size() >= 1) {
		codeName = firstDataLine[0];
	}
	else {
		qDebug() << "数据格式错误，无法获取股票名称";
		file.close();
		return;
	}

    // 定义存储数据的容器
    QVector<QDate> dates;
    QVector<double> opens, highs, lows, closes, precloses, volumes, amounts;
    // 解析数据
    while (!in.atEnd()) {
        QStringList line = in.readLine().split(",");
        if (line.size() < 9) continue; // 确保数据完整

        QDate date = QDate::fromString(line[1], "yyyy-MM-dd");
        if (!date.isValid()) continue;

        dates.append(date);
        opens.append(line[3].toDouble());
        highs.append(line[4].toDouble());
        lows.append(line[5].toDouble());
        closes.append(line[6].toDouble());
        precloses.append(line[7].toDouble());
        volumes.append(line[8].toDouble());
        amounts.append(line[9].toDouble());
    }
    file.close();
    // 将日期转换为数值（天数）
    QVector<double> x_values;
    QDate start_date = dates.first();
    for (const QDate& date : dates) {
        x_values.append(start_date.daysTo(date));
    }

    // 对每个列进行线性回归
    auto linearRegression = [](const QVector<double>& x, const QVector<double>& y, double& a, double& b) {
        int n = x.size();
        if (n != y.size() || n == 0) {
            a = b = 0;
            return;
        }
        double sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;
        for (int i = 0; i < n; ++i) {
            sum_x += x[i];
            sum_y += y[i];
            sum_xy += x[i] * y[i];
            sum_xx += x[i] * x[i];
        }
        double denominator = n * sum_xx - sum_x * sum_x;
        if (denominator == 0) {
            a = b = 0;
            return;
        }
        a = (n * sum_xy - sum_x * sum_y) / denominator;
        b = (sum_y * sum_xx - sum_x * sum_xy) / denominator;
        };

    double a_open, b_open;
    double a_high, b_high;
    double a_low, b_low;
    double a_close, b_close;
    double a_preclose, b_preclose;
    double a_volume, b_volume;
    double a_amount, b_amount;

    linearRegression(x_values, opens, a_open, b_open);
    linearRegression(x_values, highs, a_high, b_high);
    linearRegression(x_values, lows, a_low, b_low);
    linearRegression(x_values, closes, a_close, b_close);
    linearRegression(x_values, precloses, a_preclose, b_preclose);
    linearRegression(x_values, volumes, a_volume, b_volume);
    linearRegression(x_values, amounts, a_amount, b_amount);

    // 计算后三个交易日的日期（排除周末）
    QVector<QDate> predict_dates;
    QDate last_date = dates.last();
    while (predict_dates.size() < 3) {
        last_date = last_date.addDays(1);
        if (last_date.dayOfWeek() >= 1 && last_date.dayOfWeek() <= 5) { // 周一到周五
            predict_dates.append(last_date);
        }
    }

    // 将预测值写入文件
    if (!file.open(QIODevice::Append | QIODevice::Text)) {
        qDebug() << "无法打开文件进行写入";
        return;
    }
    QTextStream out(&file);

    for (const QDate& date : predict_dates) {
        double x = start_date.daysTo(date);
        QString date_str = date.toString("yyyy-MM-dd") + "<#Predict>";
        //QString code = "sh.600007"; // 假设股票代码不变

        double open_pred = a_open * x + b_open;
        double high_pred = a_high * x + b_high;
        double low_pred = a_low * x + b_low;
        double close_pred = a_close * x + b_close;
        double preclose_pred = a_preclose * x + b_preclose;
        double volume_pred = a_volume * x + b_volume;
        double amount_pred = a_amount * x + b_amount;

        QStringList data;
		data << codeName;
        data << date_str;
        data << code;
        data << QString::number(open_pred, 'f', 4);
        data << QString::number(high_pred, 'f', 4);
        data << QString::number(low_pred, 'f', 4);
        data << QString::number(close_pred, 'f', 4);
        data << QString::number(preclose_pred, 'f', 4);
        data << QString::number(volume_pred, 'f', 0);
        data << QString::number(amount_pred, 'f', 4);

        // 补充剩余列，填充为#predict
        int total_columns = headers.size();
        for (int i = data.size(); i < total_columns; ++i) {
            data << "#predict";
        }
        out << data.join(",") << "\n";
    }
    file.close();

    // 读取更新的数据并显示
    readDataFromFile(filename);
    ui->labelOutput->setText("线性回归预测完成！");
    ui->tableView->scrollToBottom();
}
void Widget::on_pushButton_predictByAI_clicked()
{
	QMessageBox::information(this, "AI预测", "AI预测功能暂未开放，敬请期待");
}


