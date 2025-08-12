

module dpram
#(parameter WIDTH = 16 , 
  parameter DEPTH = 512, 
  parameter Sel   = 0)(
    input  wire clk,
    input  wire ena,
    input  wire enb,
    input  wire wea,
    input  wire web, //1:�?  0: �?
    input  wire [$clog2(DEPTH)-1:0] addra , addrb,
    input  wire [WIDTH-1:0] dina  , dinb,
    output reg  [WIDTH-1:0] douta , doutb
);

//初始化
    reg [WIDTH-1:0] douta_temp,doutb_temp;
    reg [WIDTH-1:0] mem [DEPTH-1:0];

    always@( posedge clk ) begin : PortA
        if(ena)
        begin
             if(wea) mem[addra] <= dina;
             else begin
                douta_temp <= mem[addra];
             end
        end
    end

    always@( posedge clk ) begin : PortB
        if(enb)
        begin
             if(web) mem[addrb] <= dinb;
             else begin
                doutb_temp <= mem[addrb];
             end
        end
    end

    always@(posedge clk) begin
        douta <= douta_temp;
        doutb <= doutb_temp;
    end

    initial begin
        if(Sel == 0) $readmemh("D:/PQC/sakura/PWM_Pipeline_4/u_InitData_4.txt",mem);
        else if(Sel == 1) $readmemh("D:/PQC/sakura/PWM_Pipeline_4/s_InitData_4.txt",mem);
    end

endmodule
    
