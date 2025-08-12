module a_b_mul (
    input [11:0]a,
    input [11:0]b,
    input clk ,
    input rst_n ,
    input  start 
);
//cnt信号
    reg[1:0]cnt;
    always @( posedge clk or negedge rst_n) begin 
        if(!rst_n) cnt <= 0;
        else if (working_flag) begin
            cnt<=cnt+1'b1;
        end
    end
//working_flag的产生
    reg start_prev; // 保存前一个时刻的start信号状态
    reg working_flag; // 工作标志，表示模块是否开始工作
    always @( posedge clk or negedge rst_n) begin 
        if(!rst_n) start_prev<=0;
        else start_prev <= start; // 记录前一个时刻的start信号状态
    end
    always @( posedge clk or negedge rst_n) begin 
        if(!rst_n) working_flag <= 0;
        else if (start == 1 && start_prev == 0) begin
        // 在检测到上升沿时开始工作
        // 这里可以添加模块开始工作时的逻辑
        working_flag <= 1;
        end
    end
//求b*wn
    wire [11:0] wn;
    assign wn = 12'b0000_0000_0001;
    reg[11:0] product;
    // 第一级
    reg [23:0] mul_reg; //  b*wn
    always @( posedge clk or negedge rst_n) begin 
        if(!rst_n) mul_reg<=0;
        else if(working_flag&&(cnt=2'b0))  mul_reg <=b*wn;
        else mul_reg<=0;
    end
    //第二级
    wire [3:0] mul_h , mul_m, mul_l;
    wire [11:0] mul_q;
    assign {mul_h, mul_m, mul_l, mul_q} = mul_reg;
    wire [11:0] lut_h, lut_m,lut_l;
    wire [12:0] add13_sum0,add13_sum1;
    wire [13:0] add14_sum;
    ROM_H rom0 (
    .a (mul_h),
    .spo (lut_h)
    );
    ROM_M rom1 (
    .a (mul_m),
    .spo (lut_m)
    );

    ROM_L rom2 (
    .a (mul_l),
    .spo (lut_l)
    );

    assign add13_sum0 = lut_h + lut_m;
    assign add13_sum1 = lut_l + mul_q;
    assign add14_sum   = add13_sum0 + add13_sum1;
    reg [13:0] add14_sum_reg;
    always @( posedge clk or negedge rst_n) begin 
        if(!rst_n) add14_sum_reg<=0;
        else if(working_flag&&(cnt=2'b01))  add14_sum_reg<=add14_sum;
        else    add14_sum_reg<=0;
    end
    //第三级
    reg [11:0] ma0_j;
    wire flag;
    wire [11:0] sub,ma0_k,ma0_s;
    always@(add14_sum_reg)
      begin
        case(add14_sum_reg[13:12])
    	    2'b00: ma0_j = 12'd0;
    	    2'b01: ma0_j = 12'd767;
    	    2'b10: ma0_j = 12'd1534;
    	    2'b11: ma0_j = 12'd2301;		
    	endcase
      end

    assign {flag,sub} = add14_sum_reg[11:0] - 12'd3329;
    assign ma0_k = flag? add14_sum_reg[11:0] : sub;
    //下面是模加器
    parameter CSA_q_inv= 12'd767;  //1_0000_0000_0000-14'd3329
    wire [11:0] CSA_cin0;
    wire [11:0] CSA_cin;
    wire [11:0] CSA_sum0;
    wire [11:0] s0;
    wire [11:0] s1;
    wire sel0;
    wire sel;
    wire default_sel;// useless
    assign CSA_cin = {CSA_cin0[10:0],1'b0};
    assign sel = sel0 | CSA_cin0[11];
    assign ma0_s = sel? s0 : s1;
    assign CSA_sum0= ma0_j ^ ma0_k ^ CSA_q_inv;
    assign CSA_cin0= (ma0_j & ma0_k) | (ma0_k & CSA_q_inv) | (CSA_q_inv & ma0_j);
    assign {sel0,s0} = CSA_cin + CSA_sum0;
    assign {default_sel,s1} = ma0_j + ma0_k;
    //上面是模加器
    always @( posedge clk or negedge rst_n) begin
        if(!rst_n) product<=0;
        else if(working_flag&&(cnt=2'b10))  product<=ma0_s;
        else product<=0;
    end
//求a+b*wn和a-b*wn
    reg [11:0]a_1;//a经过一个寄存器之后
    reg [11:0]a_2;
    reg [11:0]a_3;
    always @( posedge clk or negedge rst_n) begin 
        if(!rst_n) a_1<=0;
        else if(working_flag&&(cnt=2'b00))  a_1 <=a;
        else  a_1<=0;
    end
    always @( posedge clk or negedge rst_n) begin 
        if(!rst_n) a_2<=0;
        else if(working_flag&&(cnt=2'b01))  a_2 <=a_1;
        else  a_2<=0;
    end  
    always @( posedge clk or negedge rst_n) begin 
        if(!rst_n) a_3<=0;
        else if(working_flag&&(cnt=2'b10))  a_3 <=a_2;
        else  a_3<=0;
    end
//第四级
    reg [12:0] c;//a+b*wn
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) c<=0;
        else if(working_flag&&(a_3+product>=12'd3329)&&(cnt=2'b11))c<=a_3+product-12'd3329;
        else if(working_flag&&(cnt=2'b11)) c<=a_3+product;
        else c<=0;
    end
    reg [12:0] d;//a-b*wn
    always @(*) begin
        if(!rst_n) d<=0;
        else if(working_flag&&(cnt=2'b11)&&(a_3>=product))   d<=a_3-product;
        else if(working_flag&&(cnt=2'b11)) d<=12'd3329-product+a_3;
        else d<=0;
    end

endmodule