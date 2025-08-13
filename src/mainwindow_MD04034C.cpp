#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "visa.h"
#include "visatype.h"
#include <QTcpServer>
#include <QTcpSocket>
#include<QTime>
#include<QDate>
#include <windows.h>
#include <cstdlib>
#include <ctime>
char Time[] = "HORIZONTAL:SCALE 4E-5";                   //时间40us
char Trig_Source[] ="TRIGger:A:EDGE:SOUrce CH1";         //触发源CH1
char Trig_Vol[]    = "TRIGger:A:LEVel:CH1 1.4";          //触发电平
char Trig_Mode[]   = "TRIGger:A:EDGE:SLOpe RISe";        //触发方式 rise  fAll
char Read_CH[] = "DATa:SOUrce CH2";                  //读取通道
char Data_Start[] = "DATa:START 90000";                  //start 100k - 10k
char Data_Stop [32]                                                                                                                   = "DATa:STOP 430000";                //Stop 100k + 330k
char Data_Mode[]  = "DATa:ENCdg RPBinary";           //无符号数

char PWM4_Time[] ="HORIZONTAL:SCALE 4E-7";                   //时间400ns
char PWM4_Trig_Mode[]   = "TRIGger:A:EDGE:SLOpe fAll";        //触发方式 rise  fAll
char PWM4_Data_Start[] = "DATa:START 5000";                  //start 50k
char PWM4_Data_Stop [32] = "DATa:STOP 6000";                 //Stop 60k


char PWM_Template_Time[] ="HORIZONTAL:SCALE 4E-4";                   //时间400us
char PWM_Template_Trig_Mode[]   = "TRIGger:A:EDGE:SLOpe fAll";        //触发方式   fAll
char PWM_Template_Data_Start[32] =   "DATa:START 1000000";                  //start 1m
char PWM_Template_Data_Stop [32] = "DATa:STOP 9322500";                 //Stop 60k  //2500点一个运算 取 790-840

char NTT_Template_Time[] ="HORIZONTAL:SCALE 4E-7";                   //时间400Ns
char NTT_Template_Trig_Mode[]   = "TRIGger:A:EDGE:SLOpe fAll";        //触发方式   fAll
char NTT_Template_Data_Start[32] =   "DATa:START 5000";              //start 50K
char NTT_Template_Data_Stop [32] = "DATa:STOP 6000";                 //Stop 60k

char PWM_Pipeline_Time[] ="HORIZONTAL:SCALE 4E-4";                   //时间400us
char PWM_Pipeline_Trig_Mode[]   = "TRIGger:A:EDGE:SLOpe fAll";        //触发方式   fAll
char PWM_Pipeline_Data_Start[32] =   "DATa:START 1000000";              //start 1m
char PWM_Pipeline_Data_Stop [32] = "DATa:STOP 9000000";                 //Stop 9m


char Samper_Time[] ="HORIZONTAL:SCALE 4E-7";                   //时间400ns
char Samper_Trig_Mode[]   = "TRIGger:A:EDGE:SLOpe fAll";        //触发方式   fAll
char Samper_Data_Start[32] =   "DATa:START 5000";              //start 5k
char Samper_Data_Stop [32] = "DATa:STOP 7000";                 //Stop 7k

char NTT_Pipline_Time[] ="HORIZONTAL:SCALE 4E-7";                   //时间400ns
char NTT_Pipline_Trig_Mode[]   = "TRIGger:A:EDGE:SLOpe fAll";        //触发方式   fAll
char NTT_Pipline_Data_Start[32] =   "DATa:START 2000";              //start 2k
char NTT_Pipline_Data_Stop [32] = "DATa:STOP 7000";                 //Stop 7k


char NTT_TVLA_Time[] ="HORIZONTAL:SCALE 4E-5";                   //时间40us
char NTT_TVLA_Trig_Mode[]   = "TRIGger:A:EDGE:SLOpe fAll";        //触发方式   fAll
char NTT_TVLA_Data_Start[32] =   "DATa:START 100000";              //start 100k
char NTT_TVLA_Data_Stop [32] = "DATa:STOP 1000000";                 //Stop 1000k
const int number_data = 900000;//采样点个数，stop-start



char Dili_NTT_TVLA_Time[] ="HORIZONTAL:SCALE 4E-5";                   //时间40us
char Dili_NTT_TVLA_Trig_Mode[]   = "TRIGger:A:EDGE:SLOpe fAll";        //触发方式   fAll
char Dili_NTT_TVLA_Data_Start[32] =   "DATa:START 100000";              //start 100k
char Dili_NTT_TVLA_Data_Stop [32] = "DATa:STOP 600000";                 //Stop 600k
const int Dili_NTT_Sample_cnt = 500000;
const int NTT_Pipeline_capture_count = 1;  // NTT Pipeline取均值的分母


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

//******************* tcp_server *********************//
    mServer = new QTcpServer(this);
    connect(mServer,&QTcpServer::newConnection,this,&MainWindow::NTT_Pipeline_Attack); //连接客户端
//    connect(mServer,&QTcpServer::newConnection,this,&MainWindow::Dili_NTT_TVLA_Attack); //连接客户端
//    connect(mServer,&QTcpServer::newConnection,this,&MainWindow::Sampler_Attack); //连接客户端
    //connect(mServer,&QTcpServer::newConnection,this,&MainWindow::PWM_Pipeline_Attack); //连接客户端
    //connect(mServer,&QTcpServer::newConnection,this,&MainWindow::NTT_Tamplate_Attack); //连接客户端
    //connect(mServer,&QTcpServer::newConnection,this,&MainWindow::PWM_Tamplate_Attack); //连接客户端
    //connect(mServer,&QTcpServer::newConnection,this,&MainWindow::PWM4_Attack); //连接客户端
    //connect(mServer,&QTcpServer::newConnection,this,&MainWindow::Sign_Attack); //连接客户端
    //connect(mServer,&QTcpServer::newConnection,this,&MainWindow::Sign_Attack2); //连接客户端
    //Set_Data_range(1000000,1000010);
    //启动服务器监听
    mServer->listen(QHostAddress::LocalHost,9988);
    Connected=0;
//****************************************************//



}

MainWindow::~MainWindow()
{
    delete ui;
}

void MainWindow:: Dili_NTT_TVLA_Attack()
{
    qDebug()<<"新客户段连接";
    QDateTime current_date_time =QDateTime::currentDateTime();
    QString current_date = current_date_time.toString("yyyy.MM.dd hh:mm:ss.zzz ddd");

    Connected=1;
    mSocket =new QTcpSocket(this);
    mSocket = mServer->nextPendingConnection();//与客户端通信的套接字


    uint8_t * Data_read       = new uint8_t[Dili_NTT_Sample_cnt * 2 + 10];

    int16_t * Data_Write      = new int16_t[Dili_NTT_Sample_cnt];
    int32_t * Data_Sum        = new int32_t[Dili_NTT_Sample_cnt];
    int Busy_cnt;
    int Data_locate;
    QFile f("D:\\htz3.txt");

    //******************** MDO配置 ************************//
    ViSession	rm = VI_NULL,vi = VI_NULL;
    ViUInt32	retCnt;
    char        BUSY[4] = {0,0,0,0};
    char Device[256] = "USB::0x0699::0x0456::C013077::INSTR";
    // Open a d efault Session
    viOpenDefaultRM(&rm);
    viOpen(rm, Device, VI_NULL, VI_NULL, &vi);

    //CONFIG
    viWrite(vi, (ViBuf)"ACQuire:MODe HIRes",18,&retCnt);                   //高精度模式
    viWrite(vi, (ViBuf) Dili_NTT_TVLA_Time, sizeof(Dili_NTT_TVLA_Time), &retCnt);  //时间刻度
    viWrite(vi, (ViBuf) Trig_Source, sizeof(Trig_Source), &retCnt);  //触发源
    viWrite(vi, (ViBuf) Trig_Vol, sizeof(Trig_Vol), &retCnt);        //触发电平
    viWrite(vi, (ViBuf) Dili_NTT_TVLA_Trig_Mode, sizeof(Dili_NTT_TVLA_Trig_Mode), &retCnt);      //触发方式

    viWrite(vi, (ViBuf) Read_CH, sizeof(Read_CH), &retCnt);         //读取通道
    viWrite(vi, (ViBuf) Dili_NTT_TVLA_Data_Start, sizeof(Dili_NTT_TVLA_Data_Start), &retCnt);   //起始
    viWrite(vi, (ViBuf) Dili_NTT_TVLA_Data_Stop, sizeof(Dili_NTT_TVLA_Data_Stop), &retCnt);
    viWrite(vi, (ViBuf) Data_Mode, sizeof(Data_Mode),&retCnt);      //数据模式


    viSetAttribute(vi, VI_ATTR_WR_BUF_OPER_MODE, VI_FLUSH_ON_ACCESS);
    viSetAttribute(vi, VI_ATTR_RD_BUF_OPER_MODE, VI_FLUSH_ON_ACCESS);//不用管 直接复制

    char str1[64] ="header off\n";
    char str2[64]="DATA:ENCDG RIBINARY;WIDTH 2\n"; //不用管
    viPrintf(vi, str1);
    viPrintf(vi, str2);


    QString a_Qstr;
    char u_str[10];

    QString filename = "D:\\Dili_NTT_TVLA";
    filename.append(".txt");
    f.setFileName(filename);
    if(!f.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        qDebug() << ("打开文件失败");
    }
    QTextStream txtOutput(&f);

    for(int i=0;i<100;i++){ //测10条NTT曲线
        //清空Data_sum
        for(int k=0;k<Dili_NTT_Sample_cnt;k++) Data_Sum[k] = 0;
        //测试循环 测NTT_TVLA_capture_count 次取平均值
        for(int k=0;k<NTT_TVLA_capture_count;k++)
        {
            viWrite(vi, (ViBuf) "ACQUIRE:STOPAFTER SEQUENCE", 26, &retCnt);  //Single
            BUSY[0] = '0';
            while(BUSY[0] != '1')
            {
                viWrite(vi, (ViBuf) "BUSY?", 5, &retCnt);  //等待
                viRead(vi, (ViBuf) BUSY, sizeof(BUSY), &retCnt);
                Sleep(10);
            }
            BUSY[0] = '0';
            while(BUSY[0] != '1')
            {
                viWrite(vi, (ViBuf) "BUSY?", 5, &retCnt);  //等待
                viRead(vi, (ViBuf) BUSY, sizeof(BUSY), &retCnt);
                Sleep(10);
            }

            Sleep(100);

            //strcpy(u_str,b_Qstr.toLatin1().data());
        send:
            mSocket->write("Dili_NTT1");
            mSocket->waitForBytesWritten();
            //WAIT
            //BUSY[0] = '1';
            Busy_cnt = 0;
            while(BUSY[0] != '0')
            {
                viWrite(vi, (ViBuf) "BUSY?", 5, &retCnt);  //等待
                viRead(vi, (ViBuf) BUSY, sizeof(BUSY), &retCnt);
                Sleep(10);
                Busy_cnt ++;
                if(Busy_cnt == 100) goto send;
            }
            viWrite(vi, (ViBuf) "ACQUIRE:STOPAFTER RUNSTop", 25, &retCnt);  //runstop

            //读取数据
            viWrite(vi, (ViBuf) "CURVE?", 6, &retCnt);  //获取数据
            viFlush(vi, VI_WRITE_BUF | VI_READ_BUF_DISCARD);
            viSetAttribute(vi,VI_ATTR_RD_BUF_OPER_MODE, VI_FLUSH_DISABLE);
            viRead(vi, (ViBuf) Data_read, Dili_NTT_Sample_cnt*2+10, &retCnt);
            Data_locate = Data_read[1] - '0' + 2;
            for(int k=0;k<Dili_NTT_Sample_cnt;k++)
            {
                Data_Sum[k] = (Data_read[2*k+ 1 + Data_locate] |(int16_t)(Data_read[2*k+ Data_locate] << 8)) + Data_Sum[k];
                //qDebug()<<Data_Sum[j];
            }
            viWrite(vi, (ViBuf) "ACQUIRE:STATE RUN", 17, &retCnt);  //run
        }

        //写回
        for(int k=0;k<Dili_NTT_Sample_cnt;k++)
        {
            Data_Write[k] = Data_Sum[k]/NTT_TVLA_capture_count;
            txtOutput << Data_Write[k] << endl;
        }
    }

    current_date_time =QDateTime::currentDateTime();
    current_date = current_date_time.toString("dd hh:mm:ss.zzz ddd");
    f.close();
    qDebug()<<"over";
}




void MainWindow:: NTT_TVLA_Attack()
{
    qDebug()<<"新客户段连接";
    QDateTime current_date_time =QDateTime::currentDateTime();
    QString current_date = current_date_time.toString("yyyy.MM.dd hh:mm:ss.zzz ddd");

    Connected=1;
    mSocket =new QTcpSocket(this);
    mSocket = mServer->nextPendingConnection();//与客户端通信的套接字


    uint8_t * Data_read       = new uint8_t[number_data * 2 + 10];

    int16_t * Data_Write      = new int16_t[number_data];
    int32_t * Data_Sum        = new int32_t[number_data];
    int Busy_cnt;
    int Data_locate;
    QFile f("D:\\htz3.txt");

    //******************** MDO配置 ************************//
    ViSession	rm = VI_NULL,vi = VI_NULL;
    ViUInt32	retCnt;
    char        BUSY[4] = {0,0,0,0};
    char Device[256] = "USB::0x0699::0x0456::C013077::INSTR";
    // Open a d efault Session
    viOpenDefaultRM(&rm);
    viOpen(rm, Device, VI_NULL, VI_NULL, &vi);

    //CONFIG
    viWrite(vi, (ViBuf)"ACQuire:MODe HIRes",18,&retCnt);                   //高精度模式
    viWrite(vi, (ViBuf) NTT_TVLA_Time, sizeof(NTT_TVLA_Time), &retCnt);  //时间刻度
    viWrite(vi, (ViBuf) Trig_Source, sizeof(Trig_Source), &retCnt);  //触发源
    viWrite(vi, (ViBuf) Trig_Vol, sizeof(Trig_Vol), &retCnt);        //触发电平
    viWrite(vi, (ViBuf) NTT_TVLA_Trig_Mode, sizeof(NTT_TVLA_Trig_Mode), &retCnt);      //触发方式

    viWrite(vi, (ViBuf) Read_CH, sizeof(Read_CH), &retCnt);         //读取通道
    viWrite(vi, (ViBuf) NTT_TVLA_Data_Start, sizeof(NTT_TVLA_Data_Start), &retCnt);   //起始
    viWrite(vi, (ViBuf) NTT_TVLA_Data_Stop, sizeof(NTT_TVLA_Data_Stop), &retCnt);
    viWrite(vi, (ViBuf) Data_Mode, sizeof(Data_Mode),&retCnt);      //数据模式


    viSetAttribute(vi, VI_ATTR_WR_BUF_OPER_MODE, VI_FLUSH_ON_ACCESS);
    viSetAttribute(vi, VI_ATTR_RD_BUF_OPER_MODE, VI_FLUSH_ON_ACCESS);//不用管 直接复制

    char str1[64] ="header off\n";
    char str2[64]="DATA:ENCDG RIBINARY;WIDTH 2\n"; //不用管
    viPrintf(vi, str1);
    viPrintf(vi, str2);


    QString a_Qstr;
    char u_str[10];

    QString filename = "D:\\NTT_TVLA";
    filename.append(".txt");
    f.setFileName(filename);
    if(!f.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        qDebug() << ("打开文件失败");
    }
    QTextStream txtOutput(&f);

    //清空Data_sum
    for(int k=0;k<number_data;k++) Data_Sum[k] = 0;
    //测试循环
    for(int k=0;k<NTT_TVLA_capture_count;k++)
    {
        viWrite(vi, (ViBuf) "ACQUIRE:STOPAFTER SEQUENCE", 26, &retCnt);  //Single
        BUSY[0] = '0';
        while(BUSY[0] != '1')
        {
            viWrite(vi, (ViBuf) "BUSY?", 5, &retCnt);  //等待
            viRead(vi, (ViBuf) BUSY, sizeof(BUSY), &retCnt);
            Sleep(10);
        }
        BUSY[0] = '0';
        while(BUSY[0] != '1')
        {
            viWrite(vi, (ViBuf) "BUSY?", 5, &retCnt);  //等待
            viRead(vi, (ViBuf) BUSY, sizeof(BUSY), &retCnt);
            Sleep(10);
        }

        Sleep(100);

    //strcpy(u_str,b_Qstr.toLatin1().data());
    send:
        mSocket->write("NTT_TVLA");
        mSocket->waitForBytesWritten();
        //WAIT
        //BUSY[0] = '1';
        Busy_cnt = 0;
        while(BUSY[0] != '0')
        {
            viWrite(vi, (ViBuf) "BUSY?", 5, &retCnt);  //等待
            viRead(vi, (ViBuf) BUSY, sizeof(BUSY), &retCnt);
            Sleep(10);
            Busy_cnt ++;
            if(Busy_cnt == 100) goto send;
        }
        viWrite(vi, (ViBuf) "ACQUIRE:STOPAFTER RUNSTop", 25, &retCnt);  //runstop

        //读取数据
        viWrite(vi, (ViBuf) "CURVE?", 6, &retCnt);  //获取数据
        viFlush(vi, VI_WRITE_BUF | VI_READ_BUF_DISCARD);
        viSetAttribute(vi,VI_ATTR_RD_BUF_OPER_MODE, VI_FLUSH_DISABLE);
        viRead(vi, (ViBuf) Data_read, number_data*2+10, &retCnt);
        Data_locate = Data_read[1] - '0' + 2;
        for(int k=0;k<number_data;k++)
        {
            Data_Sum[k] = (Data_read[2*k+ 1 + Data_locate] |(int16_t)(Data_read[2*k+ Data_locate] << 8)) + Data_Sum[k];
            //qDebug()<<Data_Sum[j];
        }
        viWrite(vi, (ViBuf) "ACQUIRE:STATE RUN", 17, &retCnt);  //run
    }

    for(int k=0;k<number_data;k++)
    {
        Data_Write[k] = Data_Sum[k]/NTT_TVLA_capture_count;
        txtOutput << Data_Write[k] << endl;
    }
    current_date_time =QDateTime::currentDateTime();
    current_date = current_date_time.toString("dd hh:mm:ss.zzz ddd");
    f.close();

}


// ==========================
// Test 
// ==========================
void MainWindow:: NTT_Pipeline_Attack()
{
    qDebug()<<"新客户段连接";
    QDateTime current_date_time =QDateTime::currentDateTime();
    QString current_date = current_date_time.toString("yyyy.MM.dd hh:mm:ss.zzz ddd");

    Connected=1;
    mSocket =new QTcpSocket(this);
    mSocket = mServer->nextPendingConnection();//与客户端通信的套接字


    uint8_t * Data_read       = new uint8_t[5000 * 2 + 10];

    int16_t * Data_Write      = new int16_t[5000];
    int32_t * Data_Sum        = new int32_t[5000];
    int Busy_cnt;
    int Data_locate;
    QFile f("D:\\ntt_pipeline_traces.txt");

    //******************** MDO配置 ************************//
    ViSession	rm = VI_NULL,vi = VI_NULL;
    ViUInt32	retCnt;
    char        BUSY[4] = {0,0,0,0};
    char Device[256] = "USB::0x0699::0x0456::C013077::INSTR";
    // Open a d efault Session
    viOpenDefaultRM(&rm);
    viOpen(rm, Device, VI_NULL, VI_NULL, &vi);

    //CONFIG
    viWrite(vi, (ViBuf)"ACQuire:MODe HIRes",18,&retCnt);                   //高精度模式
    viWrite(vi, (ViBuf) NTT_Pipline_Time, sizeof(NTT_Pipline_Time), &retCnt);  //时间刻度
    viWrite(vi, (ViBuf) Trig_Source, sizeof(Trig_Source), &retCnt);  //触发源
    viWrite(vi, (ViBuf) Trig_Vol, sizeof(Trig_Vol), &retCnt);        //触发电平
    viWrite(vi, (ViBuf) NTT_Pipline_Trig_Mode, sizeof(NTT_Pipline_Trig_Mode), &retCnt);      //触发方式

    viWrite(vi, (ViBuf) Read_CH, sizeof(Read_CH), &retCnt);         //读取通道
    viWrite(vi, (ViBuf) NTT_Pipline_Data_Start, sizeof(NTT_Pipline_Data_Start), &retCnt);   //起始
    viWrite(vi, (ViBuf) NTT_Pipline_Data_Stop, sizeof(NTT_Pipline_Data_Stop), &retCnt);
    viWrite(vi, (ViBuf) Data_Mode, sizeof(Data_Mode),&retCnt);      //数据模式


    viSetAttribute(vi, VI_ATTR_WR_BUF_OPER_MODE, VI_FLUSH_ON_ACCESS);
    viSetAttribute(vi, VI_ATTR_RD_BUF_OPER_MODE, VI_FLUSH_ON_ACCESS);//不用管 直接复制

    char str1[64] ="header off\n";
    char str2[64]="DATA:ENCDG RIBINARY;WIDTH 2\n"; //不用管
    viPrintf(vi, str1);
    viPrintf(vi, str2);


    QString a_Qstr;
    char u_str[10];
    const int fixed_b = 1000;  // 固定的b值
    
    // 打开文件用于写入所有曲线
    if(!f.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        qDebug() << ("打开文件失败");
        return;
    }
    QTextStream txtOutput(&f);
    
    // 初始化随机数种子
    srand(time(NULL));
    
    // 采集1000条曲线
    for(int trace_count = 0; trace_count < 1000; trace_count++)
    {
        // 生成随机的a值 (0-4095, 12位)
        int random_a = rand() % 4096;

        //清空Data_sum
        for(int k=0;k<5000;k++) Data_Sum[k] = 0;
        
        //测试循环 - 对同一个a值采集多次取平均
        for(int k=0;k<NTT_Pipeline_capture_count;k++)
        {
            viWrite(vi, (ViBuf) "ACQUIRE:STOPAFTER SEQUENCE", 26, &retCnt);  //Single
            BUSY[0] = '0';
            while(BUSY[0] != '1')
            {
                viWrite(vi, (ViBuf) "BUSY?", 5, &retCnt);  //等待
                viRead(vi, (ViBuf) BUSY, sizeof(BUSY), &retCnt);
                Sleep(10);
            }
            BUSY[0] = '0';
            while(BUSY[0] != '1')
            {
                viWrite(vi, (ViBuf) "BUSY?", 5, &retCnt);  //等待
                viRead(vi, (ViBuf) BUSY, sizeof(BUSY), &retCnt);
                Sleep(10);
            }

            Sleep(100);

            // 使用随机生成的a值
            a_Qstr = QString::number(random_a, 10);
            strcpy(u_str, a_Qstr.toLatin1().data());
            
        send:
            mSocket->write("NTT_Pipe");
            mSocket->write(u_str);
            mSocket->waitForBytesWritten();
            //WAIT
            //BUSY[0] = '1';
            Busy_cnt = 0;
            while(BUSY[0] != '0')
            {
                viWrite(vi, (ViBuf) "BUSY?", 5, &retCnt);  //等待
                viRead(vi, (ViBuf) BUSY, sizeof(BUSY), &retCnt);
                Sleep(10);
                Busy_cnt ++;
                if(Busy_cnt == 100) goto send;
            }
            viWrite(vi, (ViBuf) "ACQUIRE:STOPAFTER RUNSTop", 25, &retCnt);  //runstop

            //读取数据
            viWrite(vi, (ViBuf) "CURVE?", 6, &retCnt);  //获取数据
            viFlush(vi, VI_WRITE_BUF | VI_READ_BUF_DISCARD);
            viSetAttribute(vi,VI_ATTR_RD_BUF_OPER_MODE, VI_FLUSH_DISABLE);
            viRead(vi, (ViBuf) Data_read, 5000*2+10, &retCnt);
            Data_locate = Data_read[1] - '0' + 2;
            for(int k=0;k<5000;k++)
            {
                Data_Sum[k] = (Data_read[2*k+ 1 + Data_locate] |(int16_t)(Data_read[2*k+ Data_locate] << 8)) + Data_Sum[k];
                //qDebug()<<Data_Sum[j];
            }
            viWrite(vi, (ViBuf) "ACQUIRE:STATE RUN", 17, &retCnt);  //run
        }

        // 计算平均值并写入文件，格式为: "a的十六进制值: 采集值1 采集值2 采集值3..."
        txtOutput << QString::number(random_a, 10).toUpper() << ":";
        for(int k=0;k<5000;k++)
        {
            Data_Write[k] = Data_Sum[k]/NTT_Pipeline_capture_count;
            txtOutput << " " << Data_Write[k];
        }
        txtOutput << endl;
        
        current_date_time =QDateTime::currentDateTime();
        current_date = current_date_time.toString("dd hh:mm:ss.zzz ddd");
        qDebug() << current_date;
        qDebug() << "Trace " << trace_count << ": a=" << QString::number(random_a, 16).toUpper() << ", b=" << fixed_b;
    }
    
    f.close();
    qDebug() << "NTT Pipeline Attack completed. 1000 traces saved to ntt_pipeline_traces.txt";

}



void MainWindow::Sampler_Attack()
{
    qDebug()<<"新客户段连接";
    QDateTime current_date_time =QDateTime::currentDateTime();
    QString current_date = current_date_time.toString("yyyy.MM.dd hh:mm:ss.zzz ddd");

    Connected=1;
    mSocket =new QTcpSocket(this);
    mSocket = mServer->nextPendingConnection();//与客户端通信的套接字


    Current_row =1;
    QString a_Qstr,b_Qstr;
    char u_str[10];
    QTime timedebug;

    //检测掉线信号
    myexcel = new QExcel;
    uint8_t * Data_read       = new uint8_t[2000 * 2 + 10];

    int16_t * Data_Write      = new int16_t[2000];
    int32_t * Data_Sum        = new int32_t[2000];

    int Busy_cnt;
    int Data_locate;

    QFile f("D:\\htz3.txt");

    //******************** MDO配置 ************************//
    ViSession	rm = VI_NULL,vi = VI_NULL;
    ViUInt32	retCnt;
    char        BUSY[4] = {0,0,0,0};
    char Device[256] = "USB::0x0699::0x0456::C013077::INSTR";
    // Open a d efault Session
    viOpenDefaultRM(&rm);
    viOpen(rm, Device, VI_NULL, VI_NULL, &vi);

    //CONFIG
    viWrite(vi, (ViBuf)"ACQuire:MODe HIRes",18,&retCnt);                   //高精度模式
    viWrite(vi, (ViBuf) Samper_Time, sizeof(PWM_Pipeline_Time), &retCnt);  //时间刻度
    viWrite(vi, (ViBuf) Trig_Source, sizeof(Trig_Source), &retCnt);  //触发源
    viWrite(vi, (ViBuf) Trig_Vol, sizeof(Trig_Vol), &retCnt);        //触发电平
    viWrite(vi, (ViBuf) Samper_Trig_Mode, sizeof(PWM_Pipeline_Trig_Mode), &retCnt);      //触发方式

    viWrite(vi, (ViBuf) Read_CH, sizeof(Read_CH), &retCnt);         //读取通道
    viWrite(vi, (ViBuf) Samper_Data_Start, sizeof(PWM_Pipeline_Data_Start), &retCnt);   //起始
    viWrite(vi, (ViBuf) Samper_Data_Stop, sizeof(PWM_Pipeline_Data_Stop), &retCnt);
    viWrite(vi, (ViBuf) Data_Mode, sizeof(Data_Mode),&retCnt);      //数据模式


    viSetAttribute(vi, VI_ATTR_WR_BUF_OPER_MODE, VI_FLUSH_ON_ACCESS);
    viSetAttribute(vi, VI_ATTR_RD_BUF_OPER_MODE, VI_FLUSH_ON_ACCESS);//不用管 直接复制

    char str1[64] ="header off\n";
    char str2[64]="DATA:ENCDG RIBINARY;WIDTH 2\n"; //不用管
    viPrintf(vi, str1);
    viPrintf(vi, str2);

    for(int i=0;i<64;i=i+1)
    {

        //打开文件
        QString filename = "D:\\Sampler_Data\\";
        filename.append(QString::number(i,10));
        filename.append(".txt");
        f.setFileName(filename);
        if(!f.open(QIODevice::WriteOnly | QIODevice::Text))
        {
            qDebug() << ("打开文件失败");
        }
        QTextStream txtOutput(&f);

        //清空Data_sum
        for(int k=0;k<2000;k++)
        {
            Data_Sum[k] = 0;
        }
        //测试循环
        for(int k=0;k<Sampler_capture_count;k++)
        {
            viWrite(vi, (ViBuf) "ACQUIRE:STOPAFTER SEQUENCE", 26, &retCnt);  //Single
            BUSY[0] = '0';
            while(BUSY[0] != '1')
            {
                viWrite(vi, (ViBuf) "BUSY?", 5, &retCnt);  //等待
                viRead(vi, (ViBuf) BUSY, sizeof(BUSY), &retCnt);
                Sleep(10);
            }
            BUSY[0] = '0';
            while(BUSY[0] != '1')
            {
                viWrite(vi, (ViBuf) "BUSY?", 5, &retCnt);  //等待
                viRead(vi, (ViBuf) BUSY, sizeof(BUSY), &retCnt);
                Sleep(10);
            }

            Sleep(500);

            a_Qstr = QString::number(i,10);
            strcpy(u_str,a_Qstr.toLatin1().data());
        //strcpy(u_str,b_Qstr.toLatin1().data());
        send:
            mSocket->write("Sampler");
            mSocket->write(u_str);
            mSocket->waitForBytesWritten();
            //WAIT
            //BUSY[0] = '1';
            Busy_cnt = 0;
            while(BUSY[0] != '0')
            {
                viWrite(vi, (ViBuf) "BUSY?", 5, &retCnt);  //等待
                viRead(vi, (ViBuf) BUSY, sizeof(BUSY), &retCnt);
                Sleep(10);
                Busy_cnt ++;
                if(Busy_cnt == 100) goto send;
            }
            viWrite(vi, (ViBuf) "ACQUIRE:STOPAFTER RUNSTop", 25, &retCnt);  //runstop

            //读取数据
            viWrite(vi, (ViBuf) "CURVE?", 6, &retCnt);  //获取数据
            viFlush(vi, VI_WRITE_BUF | VI_READ_BUF_DISCARD);
            viSetAttribute(vi,VI_ATTR_RD_BUF_OPER_MODE, VI_FLUSH_DISABLE); //复制的
            viRead(vi, (ViBuf) Data_read, 2000 * 2, &retCnt);
            Data_locate = Data_read[1] - '0' + 2;

            //截断提取
            for(int k=0;k<2000;k=k+1)
            {
                Data_Sum[k] +=(Data_read[2*(k)+ 1 + Data_locate] |(int16_t)(Data_read[2*(k)+ Data_locate] << 8));
            }
            viWrite(vi, (ViBuf) "ACQUIRE:STATE RUN", 17, &retCnt);  //run
        }

        for(int k=0;k<2000;k++)
        {
            txtOutput << (Data_Sum[k])/Sampler_capture_count << endl;

        }
        current_date_time =QDateTime::currentDateTime();
        current_date = current_date_time.toString("dd hh:mm:ss.zzz ddd");
        qDebug() << current_date;
        f.close();
    }

}

// ****************************************** //
//      TEST: Pipeline "Template" Attack      //
// ****************************************** //

void MainWindow::PWM_Pipeline_Attack()
{
    qDebug()<<"新客户段连接";
    QDateTime current_date_time =QDateTime::currentDateTime();
    QString current_date = current_date_time.toString("yyyy.MM.dd hh:mm:ss.zzz ddd");

    Connected=1;
    mSocket =new QTcpSocket(this);
    mSocket = mServer->nextPendingConnection();//与客户端通信的套接字


    Current_row =1;
    QString a_Qstr,b_Qstr;
    char u_str[10];
    QTime timedebug;

    //检测掉线信号
    myexcel = new QExcel;
    uint8_t * Data_read       = new uint8_t[8000000 * 2 + 10];

    int16_t * Data_Write      = new int16_t[PWM_Pipeline_onetrace];
    int32_t * Data_Sum        = new int32_t[PWM_Pipeline_onetrace * 1500 * 3];
    int32_t * Data_Sum_zero   = new int32_t[PWM_Pipeline_onetrace * 1500 * 3];
    int Busy_cnt;
    int Data_locate;

    QFile f("D:\\htz3.txt");

    //******************** MDO配置 ************************//
    ViSession	rm = VI_NULL,vi = VI_NULL;
    ViUInt32	retCnt;
    char        BUSY[4] = {0,0,0,0};
    char Device[256] = "USB::0x0699::0x0456::C013077::INSTR";
    // Open a d efault Session
    viOpenDefaultRM(&rm);
    viOpen(rm, Device, VI_NULL, VI_NULL, &vi);

    //CONFIG
    viWrite(vi, (ViBuf)"ACQuire:MODe HIRes",18,&retCnt);
    viWrite(vi, (ViBuf) PWM_Pipeline_Time, sizeof(PWM_Pipeline_Time), &retCnt);  //时间刻度
    viWrite(vi, (ViBuf) Trig_Source, sizeof(Trig_Source), &retCnt);  //触发源
    viWrite(vi, (ViBuf) Trig_Vol, sizeof(Trig_Vol), &retCnt);        //触发电平
    viWrite(vi, (ViBuf) PWM_Pipeline_Trig_Mode, sizeof(PWM_Pipeline_Trig_Mode), &retCnt);      //触发方式

    viWrite(vi, (ViBuf) Read_CH, sizeof(Read_CH), &retCnt);         //读取通道
    viWrite(vi, (ViBuf) PWM_Pipeline_Data_Start, sizeof(PWM_Pipeline_Data_Start), &retCnt);   //起始
    viWrite(vi, (ViBuf) PWM_Pipeline_Data_Stop, sizeof(PWM_Pipeline_Data_Stop), &retCnt);
    viWrite(vi, (ViBuf) Data_Mode, sizeof(Data_Mode),&retCnt);      //数据模式


    viSetAttribute(vi, VI_ATTR_WR_BUF_OPER_MODE, VI_FLUSH_ON_ACCESS);
    viSetAttribute(vi, VI_ATTR_RD_BUF_OPER_MODE, VI_FLUSH_ON_ACCESS);

    char str1[64] ="header off\n";
    char str2[64]="DATA:ENCDG RIBINARY;WIDTH 2\n";
    viPrintf(vi, str1);
    viPrintf(vi, str2);

    for(int i=0;i<3328;i=i+1)
    {

        //if(i<1680 && i!= 0) i = 1680;
        //打开文件
        QString filename = "D:\\PWM_Pipeline_Data\\";
        filename.append(QString::number(i,10));
        filename.append(".txt");
        f.setFileName(filename);
        if(!f.open(QIODevice::WriteOnly | QIODevice::Text))
        {
            qDebug() << ("打开文件失败");
        }
        QTextStream txtOutput(&f);

        //清空Data_sum
        for(int k=0;k<PWM_Pipeline_onetrace * 1500 * 3;k++)
        {
            Data_Sum[k] = 0;
            if(i==0) Data_Sum_zero[k] = 0;
        }
        //测试循环
        for(int k=0;k<PWM_Pipeline_capture_count;k++)
        {
            viWrite(vi, (ViBuf) "ACQUIRE:STOPAFTER SEQUENCE", 26, &retCnt);  //Single
            BUSY[0] = '0';
            while(BUSY[0] != '1')
            {
                viWrite(vi, (ViBuf) "BUSY?", 5, &retCnt);  //等待
                viRead(vi, (ViBuf) BUSY, sizeof(BUSY), &retCnt);
                Sleep(10);
            }
            BUSY[0] = '0';
            while(BUSY[0] != '1')
            {
                viWrite(vi, (ViBuf) "BUSY?", 5, &retCnt);  //等待
                viRead(vi, (ViBuf) BUSY, sizeof(BUSY), &retCnt);
                Sleep(10);
            }

            Sleep(500);

            a_Qstr = QString::number(i,10);
            strcpy(u_str,a_Qstr.toLatin1().data());
        //strcpy(u_str,b_Qstr.toLatin1().data());
        send:
            mSocket->write("PWM_Pipe");
            mSocket->write(u_str);
            mSocket->waitForBytesWritten();
            //WAIT
            //BUSY[0] = '1';
            Busy_cnt = 0;
            while(BUSY[0] != '0')
            {
                viWrite(vi, (ViBuf) "BUSY?", 5, &retCnt);  //等待
                viRead(vi, (ViBuf) BUSY, sizeof(BUSY), &retCnt);
                Sleep(10);
                Busy_cnt ++;
                if(Busy_cnt == 100) goto send;
            }
            viWrite(vi, (ViBuf) "ACQUIRE:STOPAFTER RUNSTop", 25, &retCnt);  //runstop

            //读取数据
            viWrite(vi, (ViBuf) "CURVE?", 6, &retCnt);  //获取数据
            viFlush(vi, VI_WRITE_BUF | VI_READ_BUF_DISCARD);
            viSetAttribute(vi,VI_ATTR_RD_BUF_OPER_MODE, VI_FLUSH_DISABLE);
            viRead(vi, (ViBuf) Data_read, 8000000 * 2, &retCnt);
            Data_locate = Data_read[1] - '0' + 2;

            //截断提取
            for(int k=0;k<1500;k=k+1) //1500段
            {
                if(i==0) {
                    for(int j=0;j<PWM_Pipeline_onetrace;j++)
                    {
                        Data_Sum_zero[3*k*PWM_Pipeline_onetrace + j] +=  //每段第一个峰
                            (Data_read[2*(8200 + j + 5000*k)+ 1 + Data_locate] |(int16_t)(Data_read[2*(8200 + j + 5000*k)+ Data_locate] << 8));

                        Data_Sum_zero[3*k*PWM_Pipeline_onetrace + PWM_Pipeline_onetrace + j]   += //每段第二个峰
                            (Data_read[2*(9050 + j + 5000*k)+ 1 + Data_locate] |(int16_t)(Data_read[2*(9050 + j + 5000*k)+ Data_locate] << 8));

                        Data_Sum_zero[3*k*PWM_Pipeline_onetrace + 2*PWM_Pipeline_onetrace + j] += //每段第三个峰
                            (Data_read[2*(9900 + j + 5000*k)+ 1 + Data_locate] |(int16_t)(Data_read[2*(9900 + j + 5000*k)+ Data_locate] << 8));
                    }
                }
                else
                    for(int j=0;j<PWM_Pipeline_onetrace;j++)
                    {
                        Data_Sum[3*k*PWM_Pipeline_onetrace + j] +=  //每段第一个峰
                            (Data_read[2*(8200 + j + 5000*k)+ 1 + Data_locate] |(int16_t)(Data_read[2*(8200 + j + 5000*k)+ Data_locate] << 8));

                        Data_Sum[3*k*PWM_Pipeline_onetrace + PWM_Pipeline_onetrace + j]   += //每段第二个峰
                            (Data_read[2*(9050 + j + 5000*k)+ 1 + Data_locate] |(int16_t)(Data_read[2*(9050 + j + 5000*k)+ Data_locate] << 8));

                        Data_Sum[3*k*PWM_Pipeline_onetrace + 2*PWM_Pipeline_onetrace + j] += //每段第三个峰
                            (Data_read[2*(9900 + j + 5000*k)+ 1 + Data_locate] |(int16_t)(Data_read[2*(9900 + j + 5000*k)+ Data_locate] << 8));
                    }
            }
            viWrite(vi, (ViBuf) "ACQUIRE:STATE RUN", 17, &retCnt);  //run
        }

        if(i != 0)
        {
            for(int k=0;k<1500 * PWM_Pipeline_onetrace * 3;k++)
            {
                //Data_Write[k] = (Data_Sum1[k] - Data_Sum2[k])/PWM_Pipeline_capture_count;
                txtOutput << (Data_Sum[k] - Data_Sum_zero[k])/PWM_Pipeline_capture_count << endl;
                //            txtOutput0 <<Data_Sum1[k]/PWM_Pipeline_capture_count <<endl;
                //            txtOutput1 <<Data_Sum2[k]/PWM_Pipeline_capture_count <<endl;
            }
        }
        current_date_time =QDateTime::currentDateTime();
        current_date = current_date_time.toString("dd hh:mm:ss.zzz ddd");
        qDebug() << current_date;
        f.close();

    }

}


void MainWindow::NTT_Tamplate_Attack()
{
    qDebug()<<"新客户段连接";
    QDateTime current_date_time =QDateTime::currentDateTime();
    QString current_date = current_date_time.toString("yyyy.MM.dd hh:mm:ss.zzz ddd");

    Connected=1;
    mSocket =new QTcpSocket(this);
    mSocket = mServer->nextPendingConnection();//与客户端通信的套接字


    Current_row =1;
    QString a_Qstr,b_Qstr;
    char u_str[10];
    QTime timedebug;

    //检测掉线信号
    myexcel = new QExcel;
    uint8_t * Data_read       = new uint8_t[1000 * 2 + 10];
    int16_t * Data_Write      = new int16_t[1000];
    int32_t * Data_Sum        = new int32_t[1000];
    int Busy_cnt;
    int Data_locate;

    QFile f("D:\\htz3.txt");

    //******************** MDO配置 ************************//
    ViSession	rm = VI_NULL,vi = VI_NULL;
    ViUInt32	retCnt;
    char        BUSY[4] = {0,0,0,0};
    char Device[256] = "USB::0x0699::0x0456::C013077::INSTR";
    // Open a d efault Session
    viOpenDefaultRM(&rm);
    viOpen(rm, Device, VI_NULL, VI_NULL, &vi);

    //CONFIG
    viWrite(vi, (ViBuf)"ACQuire:MODe HIRes",18,&retCnt);
    viWrite(vi, (ViBuf) NTT_Template_Time, sizeof(NTT_Template_Time), &retCnt);  //时间刻度
    viWrite(vi, (ViBuf) Trig_Source, sizeof(Trig_Source), &retCnt);  //触发源
    viWrite(vi, (ViBuf) Trig_Vol, sizeof(Trig_Vol), &retCnt);        //触发电平
    viWrite(vi, (ViBuf) NTT_Template_Trig_Mode, sizeof(NTT_Template_Trig_Mode), &retCnt);      //触发方式

    viWrite(vi, (ViBuf) Read_CH, sizeof(Read_CH), &retCnt);         //读取通道
    viWrite(vi, (ViBuf) NTT_Template_Data_Start, sizeof(NTT_Template_Data_Start), &retCnt);   //起始
    viWrite(vi, (ViBuf) NTT_Template_Data_Stop, sizeof(NTT_Template_Data_Stop), &retCnt);
    viWrite(vi, (ViBuf) Data_Mode, sizeof(Data_Mode),&retCnt);      //数据模式


    viSetAttribute(vi, VI_ATTR_WR_BUF_OPER_MODE, VI_FLUSH_ON_ACCESS);
    viSetAttribute(vi, VI_ATTR_RD_BUF_OPER_MODE, VI_FLUSH_ON_ACCESS);

    char str1[64] ="header off\n";
    char str2[64]="DATA:ENCDG RIBINARY;WIDTH 2\n";
    viPrintf(vi, str1);
    viPrintf(vi, str2);

    for(int i=0;i<5;i++)
    {
        for(int j=0;j<5;j++)
        {
            QString filename = "D:\\NTT_Tamplate_Data\\";
            filename.append(QString::number((i-2 +3329)%3329,10));
            filename.append("-");
            filename.append(QString::number((j-2 + 3329)%3329,10));
            filename.append(".txt");
            f.setFileName(filename);
            if(!f.open(QIODevice::WriteOnly | QIODevice::Text))
            {
                qDebug() << ("打开文件失败");
            }
            QTextStream txtOutput(&f);

            //清空Data_sum
            for(int k=0;k<1000;k++) Data_Sum[k] = 0;
            //测试循环
            for(int k=0;k<NTT_Tamplate_capture_count;k++)
            {
                viWrite(vi, (ViBuf) "ACQUIRE:STOPAFTER SEQUENCE", 26, &retCnt);  //Single
                BUSY[0] = '0';
                while(BUSY[0] != '1')
                {
                    viWrite(vi, (ViBuf) "BUSY?", 5, &retCnt);  //等待
                    viRead(vi, (ViBuf) BUSY, sizeof(BUSY), &retCnt);
                    Sleep(10);
                }
                BUSY[0] = '0';
                while(BUSY[0] != '1')
                {
                    viWrite(vi, (ViBuf) "BUSY?", 5, &retCnt);  //等待
                    viRead(vi, (ViBuf) BUSY, sizeof(BUSY), &retCnt);
                    Sleep(10);
                }

                Sleep(100);


                a_Qstr = QString::number(5*i+j,10);
                //b_Qstr = QString::number((j-2)%3329,10);
                strcpy(u_str,a_Qstr.toLatin1().data());
                //strcpy(u_str,b_Qstr.toLatin1().data());
                send:
                mSocket->write("NTT_Tamp");
                mSocket->write(u_str);
                mSocket->waitForBytesWritten();
                //WAIT
                //BUSY[0] = '1';
                Busy_cnt = 0;
                while(BUSY[0] != '0')
                {
                    viWrite(vi, (ViBuf) "BUSY?", 5, &retCnt);  //等待
                    viRead(vi, (ViBuf) BUSY, sizeof(BUSY), &retCnt);
                    Sleep(10);
                    Busy_cnt ++;
                    if(Busy_cnt == 100) goto send;
                }
                viWrite(vi, (ViBuf) "ACQUIRE:STOPAFTER RUNSTop", 25, &retCnt);  //runstop

                //读取数据
                viWrite(vi, (ViBuf) "CURVE?", 6, &retCnt);  //获取数据
                viFlush(vi, VI_WRITE_BUF | VI_READ_BUF_DISCARD);
                viSetAttribute(vi,VI_ATTR_RD_BUF_OPER_MODE, VI_FLUSH_DISABLE);
                viRead(vi, (ViBuf) Data_read, 2000, &retCnt);
                Data_locate = Data_read[1] - '0' + 2;
                for(int k=0;k<1000;k++)
                {
                    Data_Sum[k] = (Data_read[2*k+ 1 + Data_locate] |(int16_t)(Data_read[2*k+ Data_locate] << 8)) + Data_Sum[k];
                    //qDebug()<<Data_Sum[j];
                }
                viWrite(vi, (ViBuf) "ACQUIRE:STATE RUN", 17, &retCnt);  //run
            }

            for(int k=0;k<1000;k++)
            {
                Data_Write[k] = Data_Sum[k]/NTT_Tamplate_capture_count;
                txtOutput << Data_Write[k] << endl;
            }
            current_date_time =QDateTime::currentDateTime();
            current_date = current_date_time.toString("dd hh:mm:ss.zzz ddd");
            qDebug() << current_date;
            qDebug()<<(i-2)%3329 <<" " <<(j-2)%3329<<endl;
            f.close();
        }
    }
}


void MainWindow::Set_Data_range(int Data_Start,int Data_End)
{
    QString Data_Start_str = "DATa:START ";
    QString Data_End_str = "DATa:STOP ";
    Data_Start_str += QString::number(Data_Start,10);
    Data_End_str += QString::number(Data_End,10);

    strcpy(PWM_Template_Data_Start,Data_Start_str.toLatin1().data());
    strcpy(PWM_Template_Data_Stop,Data_End_str.toLatin1().data());
}
void MainWindow::PWM_Tamplate_Attack()
{
    qDebug()<<"新客户段连接";
    QDateTime current_date_time =QDateTime::currentDateTime();
    QString current_date = current_date_time.toString("yyyy.MM.dd hh:mm:ss.zzz ddd");

    Connected=1;
    mSocket =new QTcpSocket(this);
    mSocket = mServer->nextPendingConnection();//与客户端通信的套接字


    Current_row =1;
    QString u_Qstr;
    char u_str[10];
    QTime timedebug;

    //检测掉线信号
    myexcel = new QExcel;
    uint8_t * Data_read       = new uint8_t[8322500 * 2 + 10];
    int16_t * Data_Write      = new int16_t[PWM_Tamplate_onetrace];
    int32_t * Data_Sum        = new int32_t[PWM_Tamplate_onetrace*3328];
    int Busy_cnt;
    int Data_locate;

    QFile f("D:\\htz3.txt");
//    if(!f.open(QIODevice::WriteOnly | QIODevice::Text))
//    {
//        qDebug() << ("打开文件失败");
//    }
//    QTextStream txtOutput(&f);


//    QString File = "D:/htz3.xlsx";
//    qDebug()<<File;
//    File.replace("/","\\");
//    qDebug()<<File;
//    QFile exfile(File);
//    if(exfile.exists())  myexcel->open(File); //打开一个Excel文件
//    else                 myexcel->createFile(File); //创建一个Excel文件
    //******************** MDO配置 ************************//
    ViSession	rm = VI_NULL,vi = VI_NULL;
    ViUInt32	retCnt;
    char        BUSY[4] = {0,0,0,0};
    char Device[256] = "USB::0x0699::0x0456::C013077::INSTR";
    // Open a d efault Session
    viOpenDefaultRM(&rm);
    viOpen(rm, Device, VI_NULL, VI_NULL, &vi);

    //CONFIG
    viWrite(vi, (ViBuf)"ACQuire:MODe HIRes",18,&retCnt);
    viWrite(vi, (ViBuf) PWM_Template_Time, sizeof(PWM_Template_Time), &retCnt);  //时间刻度
    viWrite(vi, (ViBuf) Trig_Source, sizeof(Trig_Source), &retCnt);  //触发源
    viWrite(vi, (ViBuf) Trig_Vol, sizeof(Trig_Vol), &retCnt);        //触发电平
    viWrite(vi, (ViBuf) PWM_Template_Trig_Mode, sizeof(PWM_Template_Trig_Mode), &retCnt);      //触发方式

    viWrite(vi, (ViBuf) Read_CH, sizeof(Read_CH), &retCnt);         //读取通道
    viWrite(vi, (ViBuf) PWM_Template_Data_Start, sizeof(PWM_Template_Data_Start), &retCnt);   //起始
    viWrite(vi, (ViBuf) PWM_Template_Data_Stop, sizeof(PWM_Template_Data_Stop), &retCnt);
    viWrite(vi, (ViBuf) Data_Mode, sizeof(Data_Mode),&retCnt);      //数据模式


    viSetAttribute(vi, VI_ATTR_WR_BUF_OPER_MODE, VI_FLUSH_ON_ACCESS);
    viSetAttribute(vi, VI_ATTR_RD_BUF_OPER_MODE, VI_FLUSH_ON_ACCESS);

    char str1[64] ="header off\n";
    char str2[64]="DATA:ENCDG RIBINARY;WIDTH 2\n";
    viPrintf(vi, str1);
    viPrintf(vi, str2);

    for(int u=1;u<3329;u++)
    {
        //打开文件
        QString filename = "D:\\PWM_Tamplate_Data\\";
        filename.append(QString::number(u,10));
        filename.append(".txt");
        f.setFileName(filename);
        if(!f.open(QIODevice::WriteOnly | QIODevice::Text))
        {
            qDebug() << ("打开文件失败");
        }
        QTextStream txtOutput(&f);
        //清空Data_sum
        for(int i=0;i<PWM_Tamplate_onetrace*3328;i++) Data_Sum[i] = 0;
        //测试循环
        for(int i=0;i<PWM_Tamplate_capture_count;i++)
        {
            viWrite(vi, (ViBuf) "ACQUIRE:STOPAFTER SEQUENCE", 26, &retCnt);  //Single
            BUSY[0] = '0';
            while(BUSY[0] != '1')
            {
                viWrite(vi, (ViBuf) "BUSY?", 5, &retCnt);  //等待
                viRead(vi, (ViBuf) BUSY, sizeof(BUSY), &retCnt);
                Sleep(10);
            }
            BUSY[0] = '0';
            while(BUSY[0] != '1')
            {
                viWrite(vi, (ViBuf) "BUSY?", 5, &retCnt);  //等待
                viRead(vi, (ViBuf) BUSY, sizeof(BUSY), &retCnt);
                Sleep(10);
            }

            Sleep(100);

        send:
            u_Qstr = QString::number(u,10);
            strcpy(u_str,u_Qstr.toLatin1().data());
            mSocket->write("PWM_Tamp");
            mSocket->write(u_str);
            mSocket->waitForBytesWritten();
            //WAIT
            //BUSY[0] = '1';
            Busy_cnt = 0;
            while(BUSY[0] != '0')
            {
                viWrite(vi, (ViBuf) "BUSY?", 5, &retCnt);  //等待
                viRead(vi, (ViBuf) BUSY, sizeof(BUSY), &retCnt);
                Sleep(10);
                Busy_cnt ++;
                if(Busy_cnt == 100) goto send;
            }
            viWrite(vi, (ViBuf) "ACQUIRE:STOPAFTER RUNSTop", 25, &retCnt);  //runstop

            //读取数据
            viWrite(vi, (ViBuf) "CURVE?", 6, &retCnt);  //获取数据
            viFlush(vi, VI_WRITE_BUF | VI_READ_BUF_DISCARD);
            viSetAttribute(vi,VI_ATTR_RD_BUF_OPER_MODE, VI_FLUSH_DISABLE);
            viRead(vi, (ViBuf) Data_read, 16645000, &retCnt);
            Data_locate = Data_read[1] - '0' + 2;
            for(int j=0;j<3328;j++)
            {
                for(int k=0 ;k<PWM_Tamplate_onetrace;k++)
                {
                    Data_Sum[PWM_Tamplate_onetrace*j + k] = (Data_read[2*(2500*j + 780 +k) + 1 + Data_locate] |(int16_t)(Data_read[2*(2500*j + 780 +k) + Data_locate] << 8)) + Data_Sum[PWM_Tamplate_onetrace*j + k];
                }
            }

            //qDebug()<<timedebug.elapsed();
            viWrite(vi, (ViBuf) "ACQUIRE:STATE RUN", 17, &retCnt);  //run
        }

        for(int j=0;j<3328;j++)
        {
            for(int k=0;k<PWM_Tamplate_onetrace;k++)
            {
                Data_Write[k] = Data_Sum[PWM_Tamplate_onetrace*j + k]/PWM_Tamplate_capture_count;
                txtOutput << Data_Write[k] << endl;
            }
        }
        current_date_time =QDateTime::currentDateTime();
        current_date = current_date_time.toString("dd hh:mm:ss.zzz ddd");
        qDebug() << current_date;
        qDebug()<<u<<endl;
        f.close();
    }
    //myexcel->save(); //保存文件
    myexcel->quit(); //退出Excel

}


void MainWindow::PWM4_Attack(){
    qDebug()<<"新客户段连接";
    QDateTime current_date_time =QDateTime::currentDateTime();
    QString current_date = current_date_time.toString("yyyy.MM.dd hh:mm:ss.zzz ddd");

    Connected=1;
    mSocket =new QTcpSocket(this);
    mSocket = mServer->nextPendingConnection();//与客户端通信的套接字

    //检测掉线信号
    myexcel = new QExcel;
    uint8_t * Data_read       = new uint8_t[PWM4_NoOf_Samplers * 2 + 10];
    int16_t * Data_Write      = new int16_t[PWM4_NoOf_Samplers];
    int32_t * Data_Sum        = new int32_t[PWM4_NoOf_Samplers];

    QString File = "D:/htz3.xlsx";
    qDebug()<<File;
    File.replace("/","\\");
    qDebug()<<File;
    QFile exfile(File);
    if(exfile.exists())  myexcel->open(File); //打开一个Excel文件
    else                 myexcel->createFile(File); //创建一个Excel文件

    //******************** MDO配置 ************************//
    ViSession	rm = VI_NULL,vi = VI_NULL;
    ViUInt32	retCnt;
    char        BUSY[4] = {0,0,0,0};
    char Device[256] = "USB::0x0699::0x0456::C013077::INSTR";
    // Open a d efault Session
    viOpenDefaultRM(&rm);
    viOpen(rm, Device, VI_NULL, VI_NULL, &vi);

    //CONFIG
    viWrite(vi, (ViBuf)"ACQuire:MODe HIRes",18,&retCnt);
    viWrite(vi, (ViBuf) PWM4_Time, sizeof(PWM4_Time), &retCnt);  //时间刻度
    viWrite(vi, (ViBuf) Trig_Source, sizeof(Trig_Source), &retCnt);  //触发源
    viWrite(vi, (ViBuf) Trig_Vol, sizeof(Trig_Vol), &retCnt);        //触发电平
    viWrite(vi, (ViBuf) PWM4_Trig_Mode, sizeof(PWM4_Trig_Mode), &retCnt);      //触发方式

    viWrite(vi, (ViBuf) Read_CH, sizeof(Read_CH), &retCnt);         //读取通道
    viWrite(vi, (ViBuf) PWM4_Data_Start, sizeof(PWM4_Data_Start), &retCnt);   //起始
    viWrite(vi, (ViBuf) PWM4_Data_Stop, sizeof(PWM4_Data_Stop), &retCnt);     //终止
    viWrite(vi, (ViBuf) Data_Mode, sizeof(Data_Mode),&retCnt);      //数据模式


    viSetAttribute(vi, VI_ATTR_WR_BUF_OPER_MODE, VI_FLUSH_ON_ACCESS);
    viSetAttribute(vi, VI_ATTR_RD_BUF_OPER_MODE, VI_FLUSH_ON_ACCESS);

    char str1[64] ="header off\n";
    char str2[64]="DATA:ENCDG RIBINARY;WIDTH 2\n";
    viPrintf(vi, str1);
    viPrintf(vi, str2);

    QFile file("C:/Users/ASUS-PC/Desktop/PQC/src/sakura/PWM4_DPA/DATA/u3000.txt");
    if(!file.exists())
    {
        qDebug("fuck");//<<"fuck";
    }
    file.open(QIODevice::ReadOnly);


    QByteArray str;
    int16_t u[4];
    char u_str[4][3];
    QString Qstr;
    int Data_locate;
    int Busy_cnt;
    Current_row =1;
    viWrite(vi, (ViBuf) "ACQUIRE:STATE RUN", 17, &retCnt);  //run
    for(int i=0;i<Continue_Num;i++)str = file.readLine();  //续上
    current_date_time =QDateTime::currentDateTime();
    current_date = current_date_time.toString("MM.dd hh:mm:ss.zzz ");

    qDebug() << current_date;
    for(int u_Num=0;u_Num <PWM4_M;u_Num ++)
    {
        //fgets(str,100,fp);
        str = file.readLine();
        for(int i=0;i<PWM4_NoOf_Samplers;i++)
        {
            Data_Sum[i] = 0;
        }
        for(int i=0;i<PWM4_capture_count;i++)
        {
            //测试循环
            viWrite(vi, (ViBuf) "ACQUIRE:STOPAFTER SEQUENCE", 26, &retCnt);  //Single
            BUSY[0] = '0';
            while(BUSY[0] != '1')
            {
                viWrite(vi, (ViBuf) "BUSY?", 5, &retCnt);  //等待
                viRead(vi, (ViBuf) BUSY, sizeof(BUSY), &retCnt);
                Sleep(10);
            }
            BUSY[0] = '0';
            while(BUSY[0] != '1')
            {
                viWrite(vi, (ViBuf) "BUSY?", 5, &retCnt);  //等待
                viRead(vi, (ViBuf) BUSY, sizeof(BUSY), &retCnt);
                Sleep(10);
            }

            Sleep(50);

            send:
            mSocket->write("PWM4");
            mSocket->write(str);
            mSocket->waitForBytesWritten();
            //WAIT
            //BUSY[0] = '1';
            Busy_cnt = 0;
            while(BUSY[0] != '0')
            {
                viWrite(vi, (ViBuf) "BUSY?", 5, &retCnt);  //等待
                viRead(vi, (ViBuf) BUSY, sizeof(BUSY), &retCnt);
                Sleep(10);
                Busy_cnt ++;
                if(Busy_cnt == 300) goto send;
            }
            //viWrite(vi, (ViBuf) "*WAI", 4, &retCnt);  //等待
            viWrite(vi, (ViBuf) "ACQUIRE:STOPAFTER RUNSTop", 25, &retCnt);  //runstop
            viWrite(vi, (ViBuf) "CURVE?", 6, &retCnt);  //获取数据
            viFlush(vi, VI_WRITE_BUF | VI_READ_BUF_DISCARD);
            viSetAttribute(vi,VI_ATTR_RD_BUF_OPER_MODE, VI_FLUSH_DISABLE);
            viRead(vi, (ViBuf) Data_read, NoOf_Samplers * 2, &retCnt);
            Data_locate = Data_read[1] - '0' + 2;
            for(int i=0 ;i<PWM4_NoOf_Samplers;i++)
            {
                Data_Sum[i] = (Data_read[2*i + 1 + Data_locate] |(int16_t)(Data_read[2*i + Data_locate] << 8)) + Data_Sum[i];
            }
            viWrite(vi, (ViBuf) "ACQUIRE:STATE RUN", 17, &retCnt);  //run
        }

        for(int i=0;i<PWM4_NoOf_Samplers;i++)
        {
            Data_Write[i] = Data_Sum[i]/PWM4_capture_count;
        }
        myexcel->htz_Write_row(Data_Write,Current_row,PWM4_NoOf_Samplers/10); Current_row ++;
        myexcel->save(); //保存文件
        current_date_time =QDateTime::currentDateTime();
        current_date = current_date_time.toString("dd hh:mm:ss.zzz ddd");
        qDebug() << current_date;
        qDebug()<<u_Num<<endl;
    }
    //myexcel->save(); //保存文件
    myexcel->quit(); //退出Excel
}


void MainWindow::Sign_Attack2()
{
    qDebug()<<"新客户段连接";
    Connected=1;
    mSocket =new QTcpSocket(this);
    mSocket = mServer->nextPendingConnection();//与客户端通信的套接字
    //关联接收客户端数据信号readyRead信号（客户端有数据就会发readyRead信号）
    //    connect(mSocket,&QTcpSocket::readyRead,this,&MainWindow::read_client_data);
    //检测掉线信号
    myexcel = new QExcel;
    ReadExcel = new QExcel;
    uint8_t * Data_read       = new uint8_t[NoOf_Samplers * 2 + 10];
    int16_t * Data_Write      = new int16_t[NoOf_Samplers];

    QString File = "D:/htz3.xlsx";
    qDebug()<<File;
    File.replace("/","\\");
    qDebug()<<File;
    QFile exfile(File);
    if(exfile.exists())  myexcel->open(File); //打开一个Excel文件
    else                 myexcel->createFile(File); //创建一个Excel文件

    //******************** MDO配置 ************************//
    ViSession	rm = VI_NULL,vi = VI_NULL;
    ViUInt32	retCnt;
    char        BUSY[4] = {0,0,0,0};

    int Data_locate;


    ViStatus status;


    char Device[256] = "USB::0x0699::0x0456::C013077::INSTR";
    // Open a d efault Session
    viOpenDefaultRM(&rm);
    viOpen(rm, Device, VI_NULL, VI_NULL, &vi);

    //CONFIG
    viWrite(vi, (ViBuf)"ACQuire:MODe HIRes",18,&retCnt);
    viWrite(vi, (ViBuf) Time, sizeof(Time), &retCnt);  //时间刻度
    viWrite(vi, (ViBuf) Trig_Source, sizeof(Trig_Source), &retCnt);  //触发源
    viWrite(vi, (ViBuf) Trig_Vol, sizeof(Trig_Vol), &retCnt);        //触发电平
    viWrite(vi, (ViBuf) Trig_Mode, sizeof(Trig_Mode), &retCnt);      //触发方式

    viWrite(vi, (ViBuf) Read_CH, sizeof(Read_CH), &retCnt);  //读取通道
    viWrite(vi, (ViBuf) Data_Start, sizeof(Data_Start), &retCnt);  //起始
    viWrite(vi, (ViBuf) Data_Stop, sizeof(Data_Stop), &retCnt);  //终止
    viWrite(vi, (ViBuf) Data_Mode, sizeof(Data_Mode),&retCnt);      //数据模式


    viSetAttribute(vi, VI_ATTR_WR_BUF_OPER_MODE, VI_FLUSH_ON_ACCESS);
    viSetAttribute(vi, VI_ATTR_RD_BUF_OPER_MODE, VI_FLUSH_ON_ACCESS);

    char str1[64] ="header off\n";
    char str2[64]="DATA:ENCDG RIBINARY;WIDTH 2\n";
    viPrintf(vi, str1);
    viPrintf(vi, str2);

    viWrite(vi, (ViBuf) "ACQUIRE:STOPAFTER SEQUENCE", 26, &retCnt);  //Single
    Sleep(500);
    mSocket->write("123");
    mSocket->waitForBytesWritten();
    //WAIT
    BUSY[0] = '1';
    while(BUSY[0] != '0')
    {
        viWrite(vi, (ViBuf) "BUSY?", 5, &retCnt);  //等待
        viRead(vi, (ViBuf) BUSY, sizeof(BUSY), &retCnt);
    }
    viWrite(vi, (ViBuf) "ACQUIRE:STOPAFTER RUNSTop", 25, &retCnt);  //runstop

    viWrite(vi, (ViBuf) "CURVE?", 6, &retCnt);  //获取数据
    status = viFlush(vi, VI_WRITE_BUF | VI_READ_BUF_DISCARD);
    status = viSetAttribute(vi,VI_ATTR_RD_BUF_OPER_MODE, VI_FLUSH_DISABLE);
    viRead(vi, (ViBuf) Data_read, NoOf_Samplers * 2, &retCnt);
    Data_locate = Data_read[1] - '0' + 2;
    for(int i=0 ;i<NoOf_Samplers;i++)
    {
        Data_Write[i] = (Data_read[2*i + 1 + Data_locate] |(int16_t)(Data_read[2*i + Data_locate] << 8));
    }

    QFile f("D:\\htz3.txt");
    if(!f.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        qDebug() << ("打开文件失败");
    }
    QTextStream txtOutput(&f);

    //全部差分数据写入txt
    for(int i=0;i<NoOf_Samplers;i++)
    {
        txtOutput << Data_Write[i] << endl;
    }
}


void MainWindow::Sign_Attack()
{
    qDebug()<<"新客户段连接";
    Connected=1;
    mSocket =new QTcpSocket(this);
    mSocket = mServer->nextPendingConnection();//与客户端通信的套接字
    //关联接收客户端数据信号readyRead信号（客户端有数据就会发readyRead信号）
//    connect(mSocket,&QTcpSocket::readyRead,this,&MainWindow::read_client_data);
    //检测掉线信号
    myexcel = new QExcel;
    ReadExcel = new QExcel;
    uint8_t * Data_read       = new uint8_t[NoOf_Samplers * 2 + 10];
    int16_t * Data_Write      = new int16_t[NoOf_Samplers];
    int32_t * Data_Sum        = new int32_t[NoOf_Samplers];
    //uint8_t Data_read[20000];
    //    int16_t Data     [NoOf_Samplers];
    //    int16_t Data_Write[10*1000];

    //QString path = QDir::currentPath();
    QString File = "D:/htz3.xlsx";
    qDebug()<<File;
    File.replace("/","\\");
    qDebug()<<File;
    QFile exfile(File);
    if(exfile.exists())  myexcel->open(File); //打开一个Excel文件
    else                 myexcel->createFile(File); //创建一个Excel文件



    //******************** MDO配置 ************************//
    ViSession	rm = VI_NULL,vi = VI_NULL;
    ViUInt32	retCnt;
    char        BUSY[4] = {0,0,0,0};

    int Data_locate;


    ViStatus status;


    char Device[256] = "USB::0x0699::0x0456::C013077::INSTR";
    // Open a d efault Session
    viOpenDefaultRM(&rm);
    viOpen(rm, Device, VI_NULL, VI_NULL, &vi);

    //CONFIG
    viWrite(vi, (ViBuf)"ACQuire:MODe HIRes",18,&retCnt);
    viWrite(vi, (ViBuf) Time, sizeof(Time), &retCnt);  //时间刻度
    viWrite(vi, (ViBuf) Trig_Source, sizeof(Trig_Source), &retCnt);  //触发源
    viWrite(vi, (ViBuf) Trig_Vol, sizeof(Trig_Vol), &retCnt);        //触发电平
    viWrite(vi, (ViBuf) Trig_Mode, sizeof(Trig_Mode), &retCnt);      //触发方式

    viWrite(vi, (ViBuf) Read_CH, sizeof(Read_CH), &retCnt);  //读取通道
    viWrite(vi, (ViBuf) Data_Start, sizeof(Data_Start), &retCnt);  //起始
    viWrite(vi, (ViBuf) Data_Stop, sizeof(Data_Stop), &retCnt);  //终止
    viWrite(vi, (ViBuf) Data_Mode, sizeof(Data_Mode),&retCnt);      //数据模式


    viSetAttribute(vi, VI_ATTR_WR_BUF_OPER_MODE, VI_FLUSH_ON_ACCESS);
    viSetAttribute(vi, VI_ATTR_RD_BUF_OPER_MODE, VI_FLUSH_ON_ACCESS);

    char str1[64] ="header off\n";
    char str2[64]="DATA:ENCDG RIBINARY;WIDTH 2\n";
    viPrintf(vi, str1);
    viPrintf(vi, str2);

    //测试循环
    for(int i=0;i<NoOf_Samplers;i++)
    {
        Data_Sum[i] = 0;
    }

    for(int i=0;i<Capture_count ;i++)
    {

        viWrite(vi, (ViBuf) "ACQUIRE:STOPAFTER SEQUENCE", 26, &retCnt);  //Single
        Sleep(500);
        mSocket->write("123");
        mSocket->waitForBytesWritten();
        //WAIT
        BUSY[0] = '1';
        while(BUSY[0] != '0')
        {
            viWrite(vi, (ViBuf) "BUSY?", 5, &retCnt);  //等待
            viRead(vi, (ViBuf) BUSY, sizeof(BUSY), &retCnt);
        }
        viWrite(vi, (ViBuf) "ACQUIRE:STOPAFTER RUNSTop", 25, &retCnt);  //runstop

        viWrite(vi, (ViBuf) "CURVE?", 6, &retCnt);  //获取数据
        status = viFlush(vi, VI_WRITE_BUF | VI_READ_BUF_DISCARD);
        status = viSetAttribute(vi,VI_ATTR_RD_BUF_OPER_MODE, VI_FLUSH_DISABLE);
        viRead(vi, (ViBuf) Data_read, NoOf_Samplers * 2, &retCnt);
        Data_locate = Data_read[1] - '0' + 2;
        for(int i=0 ;i<NoOf_Samplers;i++)
        {
            Data_Sum[i] = (Data_read[2*i + 1 + Data_locate] |(int16_t)(Data_read[2*i + Data_locate] << 8)) + Data_Sum[i];
        }
        viWrite(vi, (ViBuf) "ACQUIRE:STATE RUN", 17, &retCnt);  //run
    }

    Current_row =1;
    for(int i=0;i<NoOf_Samplers;i++)
    {
        Data_Write[i] = Data_Sum[i]/Capture_count;
    }
//    for(int i =0;i<NoOf_Samplers/(10*1000);i++)
//    {
//        myexcel->htz_Write_row(Data_Write + i * 10*1000,Current_row,10*1000);Current_row ++;
//    }

    //测试循环
    for(int i=0;i<NoOf_Samplers;i++)
    {
        Data_Sum[i] = 0;
    }

    for(int i=0;i<Capture_count ;i++)
    {

        viWrite(vi, (ViBuf) "ACQUIRE:STOPAFTER SEQUENCE", 26, &retCnt);  //Single
        Sleep(500);
        mSocket->write("234");
        mSocket->waitForBytesWritten();
        //WAIT
        BUSY[0] = '1';
        while(BUSY[0] != '0')
        {
            viWrite(vi, (ViBuf) "BUSY?", 5, &retCnt);  //等待
            viRead(vi, (ViBuf) BUSY, sizeof(BUSY), &retCnt);
        }
        viWrite(vi, (ViBuf) "ACQUIRE:STOPAFTER RUNSTop", 25, &retCnt);  //runstop

        viWrite(vi, (ViBuf) "CURVE?", 6, &retCnt);  //获取数据
        status = viFlush(vi, VI_WRITE_BUF | VI_READ_BUF_DISCARD);
        status = viSetAttribute(vi,VI_ATTR_RD_BUF_OPER_MODE, VI_FLUSH_DISABLE);
        viRead(vi, (ViBuf) Data_read, NoOf_Samplers * 2, &retCnt);
        Data_locate = Data_read[1] - '0' + 2;
        for(int i=0 ;i<NoOf_Samplers;i++)
        {
            Data_Sum[i] = (Data_read[2*i + 1 + Data_locate] |(int16_t)(Data_read[2*i + Data_locate] << 8)) + Data_Sum[i];
        }
        viWrite(vi, (ViBuf) "ACQUIRE:STATE RUN", 17, &retCnt);  //run
    }



    QFile f("D:\\htz3.txt");
    if(!f.open(QIODevice::WriteOnly | QIODevice::Text))
    {
        qDebug() << ("打开文件失败");
    }
    QTextStream txtOutput(&f);

    //全部差分数据写入txt
    for(int i=0;i<NoOf_Samplers;i++)
    {
        Data_Write[i] = Data_Write[i] - (Data_Sum[i]/Capture_count);
        txtOutput << Data_Write[i] << endl;
    }
    //部分差分数据写入excel
    for(int i =0;i<16;i++)
    {
        myexcel->htz_Write_row(Data_Write + 10000 + 20000 * i + 20000- 2500,Current_row,7500);Current_row ++;
    }

//    for(int i=0;i<NoOf_Samplers;i++)
//    {
//        Data_Write[i] = (Data_Sum[i]/Capture_count);
//    }
//    for(int i =0;i<NoOf_Samplers/(10*1000);i++)
//    {
//        myexcel->htz_Write_row(Data_Write + i * 10*1000,Current_row,10*1000);Current_row ++;
//    }



    myexcel->save(); //保存文件
    myexcel->quit(); //退出Excel
    f.close();
    free(Data_Write);
    free(Data_read);
    free(Data_Sum);
    //****************************************************//
    // Clean up
    viClose(vi); // Not needed, but makes things a bit more understandable
    viClose(rm);

    connect(mSocket,&QTcpSocket::disconnected,this,&MainWindow::client_dis);
}
void MainWindow::client_dis()
{
    QTcpSocket *obj = (QTcpSocket*)sender();//掉线对象
    qDebug()<<obj->peerAddress().toString();//打印出掉线对象的ip
    Connected=0;
}

