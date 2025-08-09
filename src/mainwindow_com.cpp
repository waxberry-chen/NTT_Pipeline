#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "qexcel.h"
#include <Qdir>
#include "math.h"
#include "time.h"
//#include "visa.h"
//#include "visatype.h"
extern "C" {
    #include "ps3000aApi.h"
}

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{

    //Socket
    mSocket = new QTcpSocket(this);
    mSocket->connectToHost(QHostAddress::LocalHost,9988);
    mSocket->waitForConnected();


    //************************************************************
    //ys begin
    this->hw = new HW_CONTROL();
    //ys end
    //************************************************************
    ui->setupUi(this);
    this->serialPort = new QSerialPort;
//    htz_open_pico();
    //initSerialPort();

    //Genreuslt();
    //Gen_SHA4_TVLA();
    //=================================================================


    //************************************************************
    //ys begin
    connect(ui->openCom, &QCheckBox::toggled, [=](bool checked){
        if (checked){
            this->hw->pData_In->open();
            ui->btnSend->setEnabled(true);
        }else{
            this->hw->pData_In->close();
            ui->btnSend->setEnabled(false);
            ui->openCom->setChecked(false);
        }
    });

    this->hw->pData_In->open();

    //this->hw->pData_In->write(0x0100,0x0001);
//    this->hw->pData_In->write(0x0002,0x0001);

    connect(mSocket,&QTcpSocket::readyRead,this,&MainWindow::Get_Sever_inst);

}

void MainWindow::Get_Sever_inst()
{
    int16_t s[4] = {0x001, 0x856, 0xce2, 0x08c};
    int16_t u[4] = {0};
    int16_t PWM_Tamplate_u = 0;
    int16_t NTT_Tamplate_a,NTT_Tamplate_b,NTT_data;

    QByteArray array = mSocket->readAll();
    QString    str;

//    this->hw->pData_In->write(0x0104,128); // 写a
//    this->hw->pData_In->write(0x0104,128); // 写a
    if(array.at(0) == '1')
    {
    //sleep(100);
        this->hw->pData_In->write(0x0140,0x000a);
        this->hw->pData_In->write(0x0140,0x0002);
        this->hw->pData_In->write(0x0140,0x000a);

        this->hw->pData_In->write(0x0002,0x0001); //Start FPGA
    }
    else if(array.at(0)=='2')
    {
        this->hw->pData_In->write(0x0140,0x0009);
        this->hw->pData_In->write(0x0140,0x0001);
        this->hw->pData_In->write(0x0140,0x0009);

        this->hw->pData_In->write(0x0002,0x0001);
    }
    else if(array.at(0)=='P' && array.at(1)=='W' && array.at(2)=='M' && array.at(3)=='4')
    {
        for(int i=0;i<4;i++)
        {
            str.append(array.at(4*i + 4));
            str.append(array.at(4*i + 5));
            str.append(array.at(4*i + 6));
            u[i] = str.toUInt(NULL,16);
            str.clear();
        }
        Set_PWM4(s,u);
        this->hw->pData_In->write(0x0002,0x0001);
    }
    else if(array.at(0)=='P' && array.at(1)=='W' && array.at(2)=='M' && array.at(3)=='_' &&
            array.at(4)=='T' && array.at(5)=='a' && array.at(6)=='m' && array.at(7)=='p')
    {
        int addr = 8;
        int t =array.size();
        while(addr<t)
        {
            str.append(array.at(addr)); addr ++;
        }
        PWM_Tamplate_u = str.toUInt(NULL,10);
        this->hw->pData_In->write(0x0100,PWM_Tamplate_u);
        this->hw->pData_In->write(0x0002,0x0001);
    }
    else if(array.at(0)=='N' && array.at(1)=='T' && array.at(2)=='T' && array.at(3)=='_' &&
             array.at(4)=='T' && array.at(5)=='a' && array.at(6)=='m' && array.at(7)=='p')
    {
        int addr = 8;
        int t =array.size();
        while(addr<t)
        {
            str.append(array.at(addr)); addr ++;
        }
        NTT_data = str.toUInt(NULL,10);
        NTT_Tamplate_a = ((NTT_data/5) - 2 + 3329 )%3329;
        NTT_Tamplate_b = (NTT_data%5 - 2 + 3329 )%3329;
        this->hw->pData_In->write(0x0100,NTT_Tamplate_a);
        this->hw->pData_In->write(0x0102,NTT_Tamplate_b);
        this->hw->pData_In->write(0x0002,0x0001);
        qDebug()<< NTT_Tamplate_a <<" "<<NTT_Tamplate_b<<endl;
    }
    else if(array.at(0)=='N' && array.at(1)=='T' && array.at(2)=='T' && array.at(3)=='_' &&
             array.at(4)=='P' && array.at(5)=='i' && array.at(6)=='p' && array.at(7)=='e')
    {// **************************** NTT Pipeline ******************************* //
        int addr = 8;
        int t =array.size();
        while(addr<t)
        {
            str.append(array.at(addr)); addr ++;
        }
        // 直接使用服务器发送的随机a值
        NTT_Tamplate_a = str.toUInt(NULL,10);
        // 使用固定的b值
        NTT_Tamplate_b = 1000;
        
        this->hw->pData_In->write(0x0100,NTT_Tamplate_a);
        this->hw->pData_In->write(0x0102,NTT_Tamplate_b);
        this->hw->pData_In->write(0x0002,0x0001);
        qDebug()<< "NTT_Pipe: a=" << NTT_Tamplate_a << ", b=" << NTT_Tamplate_b << endl;
    }
    else if(array.at(0)=='P' && array.at(1)=='W' && array.at(2)=='M' && array.at(3)=='_' &&
             array.at(4)=='P' && array.at(5)=='i' && array.at(6)=='p' && array.at(7)=='e')
    {
        int addr = 8;
        int t =array.size();
        while(addr<t)
        {
            str.append(array.at(addr)); addr ++;
        }
        PWM_Tamplate_u = str.toUInt(NULL,10);
        this->hw->pData_In->write(0x0100,100);//u
        this->hw->pData_In->write(0x0102,PWM_Tamplate_u);
        this->hw->pData_In->write(0x0002,0x0001);
        qDebug()<< PWM_Tamplate_u <<endl;
    }
    else if(array.at(0)=='S' && array.at(1)=='a' && array.at(2)=='m' && array.at(3)=='p' &&
             array.at(4)=='l' && array.at(5)=='e' && array.at(6)=='r')
    {
        int addr = 7;
        int t =array.size();
        while(addr<t)
        {
            str.append(array.at(addr)); addr ++;
        }
        PWM_Tamplate_u = str.toUInt(NULL,10);
        this->hw->pData_In->write(0x0100,100);//u
        this->hw->pData_In->write(0x0102,PWM_Tamplate_u);
        this->hw->pData_In->write(0x0002,0x0001);
        qDebug()<< PWM_Tamplate_u <<endl;
    }

    else if(array.at(0)=='N' && array.at(1)=='T' && array.at(2)=='T' && array.at(3)=='_' &&
             array.at(4)=='T' && array.at(5)=='V' && array.at(6)=='L'&& array.at(7)=='A')
    {
        this->hw->pData_In->write(0x0002,0x0001);
        //qDebug()<< PWM_Tamplate_u <<endl;
    }
    else if(array.at(0)=='D' && array.at(1)=='i' && array.at(2)=='l' && array.at(3)=='i' &&
             array.at(5)=='N' && array.at(6)=='T' && array.at(7)=='T')
    {
        this->hw->pData_In->write(0x0100,0x0000);
        if(array.at(8) == '0') this->hw->pData_In->write(0x0102,0x0000);
        else if(array.at(8) == '1') this->hw->pData_In->write(0x0102,0x0001);
        else if(array.at(8) == '2') this->hw->pData_In->write(0x0102,0x0002);
        else if(array.at(8) == '3') this->hw->pData_In->write(0x0102,0x0003);
        this->hw->pData_In->write(0x0002,0x0001);
        //qDebug()<< PWM_Tamplate_u <<endl;
    }
}

MainWindow::~MainWindow()
{
    delete ui;
}



void MainWindow:: Genreuslt(){
    WriteExcel = new QExcel;
    ReadExcel  = new QExcel;
    QString path = "C:\\Users\\ASUS-PC\\Desktop\\PQC\\src\\sakura\\PWM4_DPA";
    QFile Sample_File;

    QString HD_File_Path = "C:\\Users\\ASUS-PC\\Desktop\\PQC\\src\\sakura\\PWM4_DPA\\HD\\file1.txt";
    QString Sample_File_Path = path + "\\DATA\\u3000_2.xlsx";
    QString Write_File_Path = path + "\\DATA\\htz3.xlsx";  //写excel
    QFile file(Sample_File_Path);
    HD_File.setFileName(HD_File_Path);
    Sample_File.setFileName(Sample_File_Path);

    //WriteExcel->open(Write_File_Path);
    if(file.exists())
    {
        ReadExcel->open(Sample_File_Path);
    }
    else {
        qDebug()<<"1"<<endl;
    }

    QFile file2(Write_File_Path);
    if(file2.exists())
    {
        WriteExcel->open(Write_File_Path);
    }
    else {
        qDebug()<<"1"<<endl;
    }

    HD_File.open(QIODevice::ReadOnly);

    int16_t T[3000];
    int16_t HD[3000];
    double  r[3000];
    int16_t r_write[3000];

    ReadExcel->htz_Read_col2(T,4,3000,0);
    ReadExcel->quit();
    srand(10);
    for(int i=0;i<500;i++)  //私钥数
    {
        //GetH(HD,i,0);
        for(int j=0;j<3000;j++)
        {
            if(i==0){
                if(j%12==0) HD[j] = (rand() * rand() * rand()) %3329;
                else HD[j] = T[j] + (rand()*100) %3000 + 10000;
            }
            else HD[j] = (rand() * rand() * rand()) %3329;
        }
        for(int j=3;j<3000;j++)//选择明文数
        {
            r[j-3] = Calcuate_r(HD,T,j);
            if(r[j-3] <0 )r_write[j-3] = -r[j-3] *1000;
            else r_write[j-3] = r[j-3] *1000;
        }
        WriteExcel->htz_Write_row(r_write,i+1,2997);
        WriteExcel->save();
        qDebug()<<i<<endl;
    }
    WriteExcel->quit();


}

void MainWindow::Gen_SHA4_TVLA(){
    WriteExcel = new QExcel;
    QExcel * ReadExcel1  = new QExcel;
    QExcel * ReadExcel2  = new QExcel;

    QString path = "C:\\Users\\ASUS-PC\\Desktop\\PQC\\src\\sakura\\SHA3";

    QString Excel1_path = path + "\\1.xlsx";
    QString Excel2_path = path + "\\2.xlsx";
    QString Write_path = path + "\\htz.xlsx";

    QFile file1(Excel1_path);
    if(file1.exists()) ReadExcel1->open(Excel1_path);
    else{
        qDebug()<<"1"<<endl;
    }

    QFile file2(Excel2_path);
    if(file2.exists()) ReadExcel2->open(Excel2_path);
    else{
        qDebug()<<"1"<<endl;
    }

    QFile file3(Write_path);
    if(file3.exists()) WriteExcel->open(Write_path);
    else{
        qDebug()<<"1"<<endl;
    }

    int16_t a[5],b[5];
    double c[12000];
    int16_t c_write[12000];
    for(int i=0;i<12000;i++)
    {
        ReadExcel1->htz_Read_col2(a,i+1,5,0);
        ReadExcel2->htz_Read_col2(b,i+1,5,0);
        c[i] = Gen_t_value(a,b,5);
        c_write[i] = c[i] * 1000;
    }
    WriteExcel->htz_Write_row(c_write,1,12000);
    WriteExcel->save();
    WriteExcel->quit();
    ReadExcel1->quit();
    ReadExcel2->quit();



}

double MainWindow:: Gen_t_value(int16_t a[],int16_t b[], int length)
{
    int a_bar=0;
    int b_bar=0;
    for(int i=0;i<length;i++){
        a_bar = a_bar + a[i];
        b_bar = b_bar + b[i];
    }
    a_bar = a_bar / length;
    b_bar = b_bar / length;

    double h_Temp,t_Temp;
    double h_sigma,t_sigma;
    double ht_sum;
    h_sigma =0;
    t_sigma =0;
    ht_sum  =0;
    for(int i=0;i<length;i++)
    {
        h_Temp = a[i] - a_bar;
        t_Temp = b[i] - b_bar;
            //分母
        h_sigma = h_sigma + pow(h_Temp, 2);
        t_sigma = t_sigma + pow(t_Temp, 2);
    }

    return (a_bar - b_bar) / sqrt(h_sigma/length + t_sigma / length);
}


//初始化串口
bool MainWindow::initSerialPort(){
    this->serialPort->setPortName(ui->portName->currentText());
    if (!this->serialPort->open(QIODevice::ReadWrite)){
        QMessageBox::warning(NULL,"Tip",QStringLiteral("串口打开失败"));
        return false;
    }
    this->serialPort->setBaudRate(ui->baudRate->currentText().toInt());

    if (ui->dataBits->currentText().toInt() == 8){
        this->serialPort->setDataBits(QSerialPort::Data8);
    }else if (ui->dataBits->currentText().toInt() == 7){
        this->serialPort->setDataBits(QSerialPort::Data7);
    }else if (ui->dataBits->currentText().toInt() == 6){
        this->serialPort->setDataBits(QSerialPort::Data6);
    }else if (ui->dataBits->currentText().toInt() == 5){
        this->serialPort->setDataBits(QSerialPort::Data5);
    }

    if (ui->stopBits->currentText().toInt() == 1){
        this->serialPort->setStopBits(QSerialPort::OneStop);
    }else if (ui->stopBits->currentText().toInt() == 2){
        this->serialPort->setStopBits(QSerialPort::TwoStop);
    }

    //因为SAKURA那边的原C#上使用的是偶校验，因此这里将优先级改为偶校验第一
    /*if(ui->parity->currentText() == "NoParity"){
        this->serialPort->setParity(QSerialPort::NoParity);
    }else if (ui->parity->currentText() == "EvenParity"){
        this->serialPort->setParity(QSerialPort::EvenParity);
    }else if (ui->parity->currentText() == "OddParity"){
        this->serialPort->setParity(QSerialPort::OddParity);
    }*/
    if (ui->parity->currentText() == "EvenParity"){
        this->serialPort->setParity(QSerialPort::OddParity);
    }else if(ui->parity->currentText() == "OddParity"){
        this->serialPort->setParity(QSerialPort::NoParity);
    }else if (ui->parity->currentText() == "NoParity"){
        this->serialPort->setParity(QSerialPort::EvenParity);
    }

    //设置stopbits
    this->serialPort->setStopBits(QSerialPort::TwoStop);

    QList<QSerialPortInfo> serialPortInfos = QSerialPortInfo::availablePorts();
    qDebug() << "可用的串口数量：" << serialPortInfos.count();
    foreach (const QSerialPortInfo &serialPortInfo, serialPortInfos) {
                qDebug() << "串口名称：" << serialPortInfo.portName();
                qDebug() << "制造商：" << serialPortInfo.manufacturer();
                qDebug() << "描述：" << serialPortInfo.description();
                qDebug() << "系统位置：" << serialPortInfo.systemLocation();
            }

    return true;
}
//向串口发送信息
void MainWindow::sendMsg(const QString &msg){
    this->serialPort->write(QByteArray::fromHex(msg.toLatin1()));
    //ui->comLog->insertPlainText(QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss") + " [send] " + msg + "\n");
    ui->sendLog->insertPlainText(msg + "\n");
}
//接受来自串口的信息
//void MainWindow::recvMsg(){
//    QByteArray msg = this->serialPort->readAll();
//    //do something
//    //ui->revLog->insertPlainText(QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss") + " [recieve] " + msg.toHex().data() + "\n");
//    ui->revLog->insertPlainText(msg.toHex()+ "\n");
//}

void MainWindow::recvMsg(){
    //QByteArray msg = this->serialPort->readAll();
    QByteArray msg;
    msg.resize(this->hw->BYTELEN_TEST);
    for(int i = 0; i < this->hw->BYTELEN_TEST; ++i){
        msg[i] = this->hw->text_out[i];
    }

    for(int j = 0; j < this->hw->BYTELEN_TEST; ++j){
        qDebug() << msg[j];
    }

    //do something
    //ui->revLog->insertPlainText(QDateTime::currentDateTime().toString("yyyy-MM-dd hh:mm:ss") + " [recieve] " + msg.toHex().data() + "\n");
    ui->revLog->insertPlainText(msg.toHex()+ "\n");
}


void MainWindow::Excel_test(int16_t BufferA[],int16_t BufferB[],int16_t BufferC[],int16_t BufferD[])
{
    QExcel * myexcel;
    myexcel = new QExcel;
    //=================================================================
    QString path = QDir::currentPath();
    QString File = path + "/htz3.xlsx";
    qDebug()<<File;
    File.replace("/","\\");
    qDebug()<<File;
    QFile file(File);
    if(file.exists())  myexcel->open(File); //打开一个Excel文件
    else               myexcel->createFile(File); //创建一个Excel文件

    for (int i=1; i<=512; i++)
    {
        myexcel->setCellString(i,1,QString::number(BufferA[i]));
    }
    for (int i=1; i<=512; i++)
    {
        myexcel->setCellString(i,2,QString::number(BufferB[i]));
    }
//    for (int i=1; i<=10; i++)
//    {
//    for (int j=1; j<=10; j++)
//    {
//    myexcel->setCellString(i,j,QString::number(i+j));
//    }
//    }

    myexcel->save(); //保存文件
    myexcel->quit(); //退出Excel
}

void MainWindow:: Excel_write_col(int col,int16_t Buffer[],int length)
{
    QExcel * myexcel;
    myexcel = new QExcel;
    //=================================================================
    QString path = QDir::currentPath();
    QString File = path + "/htz3.xlsx";
    qDebug()<<File;
    File.replace("/","\\");
    qDebug()<<File;
    QFile file(File);
    if(file.exists())  myexcel->open(File); //打开一个Excel文件
    else               myexcel->createFile(File); //创建一个Excel文件

    for (int i=1; i<=length; i++)
    {
        myexcel->setCellString(i,col,QString::number(Buffer[i]));
    }

    myexcel->save(); //保存文件
    myexcel->quit(); //退出Excel
}

//**************************************************************************************************************
//参考：https://blog.csdn.net/qq_34342853/article/details/118705220
void MainWindow::openOSC(int ena,int enb,  int enc, int end,int index){
        chart = new QChart();
        QLineSeries * seriesA =new QLineSeries();
        QLineSeries * seriesB =new QLineSeries();
        QLineSeries * seriesC =new QLineSeries();
        QLineSeries * seriesD =new QLineSeries();

        QValueAxis *xAxis = new QValueAxis();
        QValueAxis *yAxis = new QValueAxis();


        xAxis->setRange(0, NoOf_Samples -1);
        yAxis->setRange(-32512,32512);
        chart->addAxis(yAxis, Qt::AlignLeft);
        chart->addAxis(xAxis, Qt::AlignBottom);

        xAxis->setGridLineVisible(true);
        xAxis->setGridLineColor(Qt::black);



        seriesA->setPen(QPen(QColor(0, 0, 255), 3));
        seriesB->setPen(QPen(QColor(255, 0, 0), 3));
        seriesC->setPen(QPen(QColor(0, 170, 127), 3));
        seriesD->setPen(QPen(QColor(170, 170, 0), 3));

//        ui->displayOSC->setChart(chart);
//        ui->displayOSC->resize(800,400);
        chart->removeAllSeries();
        //series = new QLineSeries();
        int counter = 0;
        if(ena){
            for (counter = 0; counter < NoOf_Samples; counter++) {
                seriesA->append(counter, BufferA[counter]);
            }

            chart->addSeries(seriesA);
            seriesA->attachAxis(xAxis);
            seriesA->attachAxis(yAxis);

        }

        counter = 0;
        if(enb){

            for (counter = 0; counter < NoOf_Samples; counter++) {
                seriesB->append(counter, BufferB[counter]);
            }

             chart->addSeries(seriesB);
             seriesB->attachAxis(xAxis);
             seriesB->attachAxis(yAxis);
        }

        counter = 0;
        if(enc){
            for (counter = 0; counter < NoOf_Samples; counter++) {
                seriesC->append(counter, BufferC[counter]);
            }
             chart->addSeries(seriesC);
//             seriesC->attachAxis(xAxis);
//             seriesC->attachAxis(yAxis);
        }

        counter = 0;
        if(end){
            for (counter = 0; counter < NoOf_Samples; counter++) {
                seriesD->append(counter, BufferD[counter]);
            }
             chart->addSeries(seriesD);
             //seriesD->attachAxis(xAxis);
             //seriesD->attachAxis(yAxis);
        }



        chart->createDefaultAxes();



        chart->setTitle("PQC之示波器");
        chart->legend()->hide(); //不然会有个点儿

        ui->displayOSC->setChart(chart);
        //ui->displayOSC->resize(800,400);

        ui->displayOSC->setRubberBand(QChartView::RectangleRubberBand);
        ui->displayOSC->setRenderHint(QPainter::Antialiasing, true);


}


//void MainWindow::Run_Pico()
//{


////    this->Set_Key();             // Shake256
////    this->Set_Wots(0,0,16,0,2);  // Shake256

//    int16_t s[4] = {0x975, 0xc46, 0xbf3, 0xacf};
//    int16_t u[4] = {0x029, 0x71e, 0xbbd, 0xc7d};
//    this->Set_PWM4(s,u);
//    //this->Set_Adder(0,0);
//    //Mode  gensk
//    //this->hw->pData_In->write(0x0150,0);
//    for(int i=0;i<Capture_count;i++)
//    {
//        Set_Pico();
//        int NoOf_Channel = A_enable + B_enable + C_enable + D_enable;
//        htz_Run_Block(Time_Base,NoOf_Samples,NoOf_Channel);
//        Sleep(1000);
//        this->hw->pData_In->write(0x0002,0x0001);
//        htz_get_data(NoOf_Samples,NoOf_Channel);
//        for(int j=0;j<NoOf_Samples;j++)
//        {
//            Buffer_Sum[0][j] = BufferB[j] +Buffer_Sum[0][j];
//        }
//        Current_index ++;
//    }
//    //this->Set_Wots(0,0,16,0,1); //Shake256
////    this->Set_Adder(767,767);
//    //u[4] = {0x9dc, 0x968, 0x5d3, 0xaa6};

//    this->Set_PWM4(s,u);
//    for(int i=0;i<Capture_count;i++)
//    {
//        Set_Pico();
//        int NoOf_Channel = A_enable + B_enable + C_enable + D_enable;
//        htz_Run_Block(Time_Base,NoOf_Samples,NoOf_Channel);
//        Sleep(1000);
//        this->hw->pData_In->write(0x0002,0x0001);
//        htz_get_data(NoOf_Samples,NoOf_Channel);
//        for(int j=0;j<NoOf_Samples;j++)
//        {
//            Buffer_Sum[1][j] = BufferB[j] +Buffer_Sum[1][j];
//        }
//    }

//    //Excel_write_col
//    QString path = QDir::currentPath();
//    QString File = path + "/htz3.xlsx";
//    qDebug()<<File;
//    File.replace("/","\\");
//    qDebug()<<File;
//    QFile file(File);
//    if(file.exists())  myexcel->open(File); //打开一个Excel文件
//    else               myexcel->createFile(File); //创建一个Excel文件


////    myexcel->htz_Write_row(BufferB[1],Current_col,NoOf_Samples);    Current_col++;
////    myexcel->htz_Write_row(BufferB[2],Current_col,NoOf_Samples);    Current_col++;
////    myexcel->htz_Write_row(BufferB[4],Current_col,NoOf_Samples);    Current_col++;
////    myexcel->htz_Write_row(BufferB[5],Current_col,NoOf_Samples);    Current_col++;
////    Current_col++;
////    //SAME1
////    for(int i=0;i<NoOf_Samples;i++)
////    {
////        Buffer[i] = BufferB[1][i] - BufferB[2][i];
////    }

////    myexcel->htz_Write_row(Buffer,Current_col,NoOf_Samples);        Current_col++;
////    //SAME2
////    for(int i=0;i<NoOf_Samples;i++)
////    {
////        Buffer[i] = BufferB[4][i] - BufferB[5][i];
////    }

////    myexcel->htz_Write_row(Buffer,Current_col,NoOf_Samples);        Current_col++;
////    Current_col++;

////    //DIFF1
////    for(int i=0;i<NoOf_Samples;i++)
////    {
////        Buffer[i] = BufferB[1][i] - BufferB[5][i];
////    }

////    myexcel->htz_Write_row(Buffer,Current_col,NoOf_Samples);        Current_col++;

////    //same2
////    for(int i=0;i<NoOf_Samples;i++)
////    {
////        Buffer[i] = BufferB[4][i] - BufferB[2][i];
////    }

////    myexcel->htz_Write_row(Buffer,Current_col,NoOf_Samples);        Current_col++;
////    Current_col++;

//    for(int i=0;i<NoOf_Samples;i++)
//    {
//        Buffer[i] = (Buffer_Sum[0][i] / Capture_count);
//    }
//    myexcel->htz_Write_row(Buffer,Current_col,NoOf_Samples);    Current_col++;

//    for(int i=0;i<NoOf_Samples;i++)
//    {
//        Buffer[i] = (Buffer_Sum[1][i] / Capture_count);
//    }
//    myexcel->htz_Write_row(Buffer,Current_col,NoOf_Samples);    Current_col++;

//    for(int i=0;i<NoOf_Samples;i++)
//    {
//        Buffer[i] = ((Buffer_Sum[1][i] - Buffer_Sum[0][i]) / Capture_count);
//    }
//    myexcel->htz_Write_row(Buffer,Current_col,NoOf_Samples);    Current_col++;


//    myexcel->save(); //保存文件
//    myexcel->quit(); //退出Excel
//    int16_t corr_bufferA[2000],corr_bufferB[2000];
//    for(int i=0;i<5;i++)
//    {
//        for(int j=0;j<2000;j++)
//        {
////            corr_bufferA[j] = Buffer_Sum[0][2048 + i*2000 +j]/ Capture_count;
////            corr_bufferB[j] = Buffer_Sum[1][2048 + i*2000 +j]/ Capture_count;
//            corr_bufferA[j] = Buffer[2048 + i*2000 +j];
//            corr_bufferB[j] = Buffer[2048 + i*2000 +j + 2000];
//        }
////        myexcel->htz_Write_row(corr_bufferA,Current_col,2000);    Current_col++;
////        myexcel->htz_Write_row(corr_bufferB,Current_col,2000);    Current_col++;
//        Current_col++;
//        corr[i] = Corrcoef(corr_bufferA,corr_bufferB,2000);
//    }
////    myexcel->save(); //保存文件
////    myexcel->quit(); //退出Excel

//}

void MainWindow::Set_Wots(int tree_index,int tree_height,int layer,int keypair,int Mode)
{
    //Tree_index
    this->hw->pData_In->write(0x0140,tree_index);

    //Tree_height
    this->hw->pData_In->write(0x0142,tree_height);

    //Tree_Keypair
    this->hw->pData_In->write(0x0144,keypair);

    //Layer
    this->hw->pData_In->write(0x0146,layer);

    //Mode
    this->hw->pData_In->write(0x0150,Mode);
}

void MainWindow::Set_Adder(int16_t a, int16_t b)
{
    this->hw->pData_In->write(0x0100,a);

    this->hw->pData_In->write(0x0102,b);
}
void MainWindow::Set_Key()
{
    uint16_t Data0 = 0x00;
    uint16_t Data1 = 0x01;
    for(int i=0;i<16;i++)
    {
        this->hw->pData_In->write(0x0100 + 2*i,(Data1 <<8 | Data0));
        Data0 = Data0 + 2;
        Data1 = Data1 + 2;
    }

    Data0 = 0x40;
    Data1 = 0x41;
    for(int i=0;i<16;i++)
    {

        this->hw->pData_In->write(0x0120 + 2*i,(Data1 <<8 | Data0));
        Data0 = Data0 + 2;
        Data1 = Data1 + 2;
    }
    //sel_tree
    Data0 = 0x00;
    Data1 = 0x00;
    for(int i=0;i<4;i++)
    {
        Data0 = Data0 + 2;
        Data1 = Data1 + 2;
        this->hw->pData_In->write(0x0148 + 2*i,0);
    }
}

//void MainWindow::Set_Pico()
//{
//     A_enable = ui->A_AC_DC->currentIndex() != 2;
//    int A_AC_DC  = ui->A_AC_DC->currentIndex() == 0;    //直流
//    int A_Range  = ui->A_Vol->currentIndex();
//    htz_Set_Channel(PS3000A_CHANNEL_A,A_enable,A_Range,A_AC_DC);

//     B_enable = ui->B_AC_DC->currentIndex() != 2;
//    int B_AC_DC  = ui->B_AC_DC->currentIndex() == 0;    //直流
//    int B_Range  = ui->B_Vol->currentIndex();
//    htz_Set_Channel(PS3000A_CHANNEL_B,B_enable,B_Range,B_AC_DC);

//     C_enable = ui->C_AC_DC->currentIndex() != 2;
//    int C_AC_DC  = ui->C_AC_DC->currentIndex() == 0;    //直流
//    int C_Range  = ui->C_Vol->currentIndex();
//    htz_Set_Channel(PS3000A_CHANNEL_C,C_enable,C_Range,C_AC_DC);

//     D_enable = ui->D_AC_DC->currentIndex() != 2;
//    int D_AC_DC  = ui->D_AC_DC->currentIndex() == 0;    //直流
//    int D_Range  = ui->D_Vol->currentIndex();
//    htz_Set_Channel(PS3000A_CHANNEL_D,D_enable,D_Range,D_AC_DC);


//    Time_Base = ui->Time->currentIndex();
//    switch (ui->Time->currentIndex()) {
//        case 0: Time_Base =2; break;
//        case 1: Time_Base =3; break;
//        case 2: Time_Base =4; break;
//        case 3: Time_Base =7; break;
//        case 4: Time_Base =12;break;
//        case 5: Time_Base =22;break;
//    default: Time_Base = 2;
//    }
//    htz_Set_Time_Base( Time_Base, NoOf_Samples);

//    int Range[4] = {A_Range,B_Range,C_Range,D_Range};
//    int Trig_Channel = ui->Trig_Channel->currentIndex();
//    int Trig_Vol     = ui->Trig_Vol->text().toUInt();
//    Trig_Vol = htz_mv_to_adc(Trig_Vol,Range[Trig_Channel]);
//    int Trig_Type    = ui->Trig_Mode->currentIndex();
//    htz_Set_Trigger(1,Trig_Channel,Trig_Vol,Trig_Type);

//    htz_Set_Buffer(BufferA,BufferB,BufferC,BufferD,
//                       A_enable,B_enable,C_enable,D_enable, NoOf_Samples);
//}

double MainWindow::Corrcoef(int16_t A[],int16_t B[], int Length)
{
    //计算相关系数
    //A		自变量A数组
    //B		自变量B数组
    //Length 	数组长度

        double sumA, sumB, aveA, aveB;
        int i;
        //求和
        sumA=0;
        for(i=0;i<Length;i++)
        {

            sumA+=A[i];
        }
        sumB=0;
        for(i=0;i<Length;i++)
        {

            sumB+=B[i];
        }

        //求平均值
        aveA = sumA / double(Length);
        aveB = sumB / double(Length);

        //计算相关系数
        double R1=0, R2=0, R3=0;
        for (long i = 0; i < Length; i++)
        {
            R1 += (A[i] - aveA) * (B[i] - aveB);
            R2 += pow((A[i] - aveA), 2);
            R3 += pow((B[i] - aveB), 2);
        }

        return (R1 / sqrt(R2*R3));

}

void MainWindow::Set_PWM4(int16_t s[],int16_t u[])
{
    this->hw->pData_In->write(0x0100,s[0]);
    this->hw->pData_In->write(0x0102,s[1]);
    this->hw->pData_In->write(0x0104,s[2]);
    this->hw->pData_In->write(0x0106,s[3]);

    this->hw->pData_In->write(0x0108,u[0]);
    this->hw->pData_In->write(0x010a,u[1]);
    this->hw->pData_In->write(0x010c,u[2]);
    this->hw->pData_In->write(0x010e,u[3]);
}


//void MainWindow::Sign_Attack()
//{
//    //Excel_write_col
//    QString path = QDir::currentPath();
//    QString File = path + "/htz3.xlsx";
//    qDebug()<<File;
//    File.replace("/","\\");
//    qDebug()<<File;
//    QFile exfile(File);
//    if(exfile.exists())  myexcel->open(File); //打开一个Excel文件
//    else               myexcel->createFile(File); //创建一个Excel文件

//    int16_t Buffer_Write[NoOf_Samples];

//    for(int i=0;i<10;i++)
//    {
//        Set_Pico();
//        int NoOf_Channel = A_enable + B_enable + C_enable + D_enable;
//        htz_Run_Block(Time_Base,NoOf_Samples,NoOf_Channel);
//        Sleep(100);
//        this->hw->pData_In->write(0x0140,0x000a);
//        this->hw->pData_In->write(0x0140,0x0002);
//        this->hw->pData_In->write(0x0140,0x000a);
//        this->hw->pData_In->write(0x0002,0x0001);
//        htz_get_data(NoOf_Samples,NoOf_Channel);
////        myexcel->htz_Write_row(BufferB,Current_col,NoOf_Samples);
////        Current_col ++;
//    }

//    for(int i=0;i<NoOf_Samples;i++) BufferB[i] = 0;


//  //*************** Sign采样 ****************//
//    for(int i=0;i<Capture_count;i++)  //Sign采样
//    {
//        Set_Pico();
//        int NoOf_Channel = A_enable + B_enable + C_enable + D_enable;
//        htz_Run_Block(Time_Base,NoOf_Samples,NoOf_Channel);
//        Sleep(10);
//        this->hw->pData_In->write(0x0140,0x000a);
//        this->hw->pData_In->write(0x0140,0x0002);
//        this->hw->pData_In->write(0x0140,0x000a);
//        this->hw->pData_In->write(0x0002,0x0001);
//        htz_get_data(NoOf_Samples,NoOf_Channel);
//        //Sleep(100);
//        for(int j=0;j<NoOf_Samples;j++)
//        {
//            Buffer_Sum[0][j] = BufferB[j] +Buffer_Sum[0][j];
//        }
//        //Current_index ++;
//    }

//    for(int i=0;i<NoOf_Samples;i++)
//    {
//        Buffer_Write[i] = Buffer_Sum[0][i] /Capture_count;
//    }
//    for(int i=0;i< NoOf_Samples / (1024 * 16);i++)
//    {
//        myexcel->htz_Write_row(Buffer_Write + 1024 * 16 * i,Current_col,NoOf_Samples/4); Current_col ++;
//    }
////    myexcel->htz_Write_row(Buffer_Write,Current_col,NoOf_Samples/4); Current_col ++;
////    myexcel->htz_Write_row(Buffer_Write + NoOf_Samples * 1/4,Current_col,NoOf_Samples/4); Current_col ++;
////    myexcel->htz_Write_row(Buffer_Write + NoOf_Samples * 2/4,Current_col,NoOf_Samples/4); Current_col ++;
////    myexcel->htz_Write_row(Buffer_Write + NoOf_Samples * 3/4,Current_col,NoOf_Samples/4); Current_col ++;
//    //*******************************************//



//    //*************** KeyGen采样 ****************//
//    for(int i=0;i<Capture_count;i++)
//    {
//        Set_Pico();
//        int NoOf_Channel = A_enable + B_enable + C_enable + D_enable;
//        htz_Run_Block(Time_Base,NoOf_Samples,NoOf_Channel);
//        Sleep(10);
//        this->hw->pData_In->write(0x0140,0x0009);
//        this->hw->pData_In->write(0x0140,0x0001);
//        this->hw->pData_In->write(0x0140,0x0009);
//        this->hw->pData_In->write(0x0002,0x0001);
//        htz_get_data(NoOf_Samples,NoOf_Channel);
//        //Sleep(100);
//        for(int j=0;j<NoOf_Samples;j++)
//        {
//            Buffer_Sum[1][j] = BufferB[j] +Buffer_Sum[1][j];
//        }
//        //Current_index ++;
//    }

//    for(int i=0;i<NoOf_Samples;i++)
//    {
//        Buffer_Write[i] = Buffer_Sum[1][i] /Capture_count;
//    }
////    myexcel->htz_Write_row(Buffer_Write,Current_col,NoOf_Samples/2); Current_col ++;
////    myexcel->htz_Write_row(Buffer_Write + NoOf_Samples/2,Current_col,NoOf_Samples/2); Current_col ++;
//    for(int i=0;i< NoOf_Samples / (1024 * 16);i++)
//    {
//        myexcel->htz_Write_row(Buffer_Write + 1024 * 16 * i,Current_col,NoOf_Samples/4); Current_col ++;
//    }
//    //*******************************************//

//    for(int i=0;i<NoOf_Samples;i++)
//    {
//        Buffer_Write[i] = (Buffer_Sum[0][i] - Buffer_Sum[1][i]) /Capture_count;
//    }
////    myexcel->htz_Write_row(Buffer_Write,Current_col,NoOf_Samples/2); Current_col ++;
////    myexcel->htz_Write_row(Buffer_Write + NoOf_Samples/2,Current_col,NoOf_Samples/2); Current_col ++;
//    for(int i=0;i< NoOf_Samples / (1024 * 16);i++)
//    {
//        myexcel->htz_Write_row(Buffer_Write + 1024 * 16 * i,Current_col,NoOf_Samples/4); Current_col ++;
//    }

//    myexcel->save(); //保存文件
//    myexcel->quit(); //退出Excel
//}

//void MainWindow::PWM4_Loop()
//{
//    QFile file("C:/Users/ASUS-PC/Desktop/PQC/src/sakura/PWM4_DPA/DATA/u3000.txt");
//    if(!file.exists())
//    {
//        qDebug("fuck");//<<"fuck";
//    }
//    file.open(QIODevice::ReadOnly);
//    //FILE *fp=fopen("‪C:/Users/ASUS-PC/Desktop/PQC/src/sakura/PWM4_DPA/DATA/u.txt","r");
//    //char str[100];
//    QByteArray str;
//    int16_t u[4];
//    int16_t s[4] = {0x975, 0xc46, 0xbf3, 0xacf};
//    char u_str[4][3];
//    QString Qstr;
//    int16_t Buffer_Write[512];


//    //Excel_write_col
//    QString path = QDir::currentPath();
//    QString File = path + "/htz3.xlsx";
//    qDebug()<<File;
//    File.replace("/","\\");
//    qDebug()<<File;
//    QFile exfile(File);
//    if(exfile.exists())  myexcel->open(File); //打开一个Excel文件
//    else               myexcel->createFile(File); //创建一个Excel文件
//    for(int i=0;i<10;i++)
//    {
//        //Set_Pico();
//        int NoOf_Channel = A_enable + B_enable + C_enable + D_enable;
//        htz_Run_Block(Time_Base,NoOf_Samples,NoOf_Channel);
//        Sleep(100);
//        this->hw->pData_In->write(0x0002,0x0001);
//        //htz_get_data(NoOf_Samples,NoOf_Channel);
//        myexcel->htz_Write_row(BufferB,Current_col,360);
//        Current_col ++;
//    }

//    for(int i=0;i<NoOf_Samples;i++) BufferB[i] = 0;

//    for(int u_Num=0;u_Num <M;u_Num ++)
//    {
//        //fgets(str,100,fp);
//        str = file.readLine();
//        int k=0;
//        for(int i=0;i<4;i++)
//        {
//            for(int j=0;j<3;j = (str.at(k-1)!=' ') ? j+1 : j)
//            {
//                if(str.at(k)!=' ')
//                {
//                    u_str[i][j] = str.at(k);
//                }
//                k++;
//            }
//        }
//        for(int i=0;i<4;i++)  //get u
//        {
//            Qstr.append(u_str[i][0]);// = u_str[i];
//            Qstr.append(u_str[i][1]);
//            Qstr.append(u_str[i][2]);
//            Qstr.append('\n');
//            u[i] = Qstr.toUInt(NULL,16);
//            Qstr.clear();
//        }

//        Set_PWM4(s,u);


//        for(int i=0;i<Capture_count;i++)  //采样
//        {
//            //Set_Pico();
//            int NoOf_Channel = A_enable + B_enable + C_enable + D_enable;
//            htz_Run_Block(Time_Base,NoOf_Samples,NoOf_Channel);
//            Sleep(10);
//            this->hw->pData_In->write(0x0002,0x0001);
//            htz_get_data(NoOf_Samples,NoOf_Channel);
//            //Sleep(100);
//            for(int j=0;j<NoOf_Samples;j++)
//            {
//                Buffer_Sum[0][j] = BufferB[j] +Buffer_Sum[0][j];
//            }
//            Current_index ++;
//        }
////        for(int i=0;i<4;i++)
////        {
////            for(int j=0;j<83;j++)
////            {
////                Buffer_Write[j] = Buffer_Sum[0][1003 + i*83 + j] /100;
////            }
////            myexcel->htz_Write_row(Buffer_Write,Current_col,83);
////            Current_col ++;
////        }
//        for(int i=0;i<360;i++)
//        {
//            Buffer_Write[i] = Buffer_Sum[0][1003 + i] /Capture_count;
//        }
//        myexcel->htz_Write_row(Buffer_Write,Current_col,360);
//        Current_col ++;
//        for(int i =0;i<NoOf_Samples;i++) {
//            Buffer_Sum[0][i]=0;
//        }
//        qDebug()<<u_Num<<'\n';
//    }
//    myexcel->save(); //保存文件
//    myexcel->quit(); //退出Excel

//}

void MainWindow::GetH(int16_t HD[],int sk_index,int coef_index)
{
    //QFile file("C:/Users/ASUS-PC/Desktop/PQC/src/sakura/PWM4_DPA/HD/hd_product.txt");

    QByteArray str;
    char H_str[4] ={0};
    //H_str[0] =0;
    QString Qstr;
    int k;
    int16_t fuck;

    //sk_index of u
    for(int i=0;i<M;i++)
    {
        str = HD_File.readLine();
        k=0;
        for(int j=0;j<3;j++)
        {
            if(str.at(4*coef_index + j) == '0' & k==0);
            else {
                H_str[k] = str.at(4*coef_index + j);
                k ++;
            }

        }
        H_str[k] = '\n';
        Qstr = H_str;
        fuck = Qstr.toInt(NULL,16);
        HD[i] = fuck;
        Qstr.clear();
    }
    //file.close();
}

//void MainWindow:: GetT(int coef_index,int Time)
//{
//    QFile file("C:/Users/ASUS-PC/Desktop/PQC/src/sakura/PWM4_DPA/DATA/1.xlsx");
//    if(file.exists())  ReadExcel->open("C:/Users/ASUS-PC/Desktop/PQC/src/sakura/PWM4_DPA/DATA/1.xlsx"); //打开一个Excel文件

//    ReadExcel->htz_Read_col(T,Time + 1,M,coef_index);
//    ReadExcel->quit();
//}
//void MainWindow:: GetT2(int coef_index,int Time)
//{
////    QFile file("C:/Users/ASUS-PC/Desktop/PQC/src/sakura/PWM4_DPA/DATA/u3000_2.xlsx");
////    if(file.exists())  ReadExcel->open("C:/Users/ASUS-PC/Desktop/PQC/src/sakura/PWM4_DPA/DATA/u3000_2.xlsx"); //打开一个Excel文件

//    ReadExcel->htz_Read_col2(T,Time + 1,M,coef_index);
//    ReadExcel->quit();
//}
double MainWindow::Calcuate_r (int16_t H[],int16_t T[], int length)
{
    double h_bar,t_bar;
    h_bar = 0; t_bar =0;
    for(int i=0;i<length;i++)
    {
        h_bar = h_bar + H[i];
        t_bar = t_bar + T[i];
    }
    h_bar = h_bar/length;
    t_bar = t_bar/length;

   double h_Temp,t_Temp;
   double h_sigma,t_sigma;
   double ht_sum;
   h_sigma =0;
   t_sigma =0;
   ht_sum  =0;
   for(int i=0;i<length;i++)
   {
       h_Temp = H[i] - h_bar;
       t_Temp = T[i] - t_bar;
        //分母
       h_sigma = h_sigma + pow(h_Temp, 2);
       t_sigma = t_sigma + pow(t_Temp, 2);

       //分子
       ht_sum = ht_sum + h_Temp * t_Temp;
   }

   double r = ht_sum / sqrt(h_sigma * t_sigma);
   return r;
}


