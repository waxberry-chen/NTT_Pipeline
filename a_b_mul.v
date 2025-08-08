module a_b_mul (
    (* dont_touch = "true" *)(* keep = "TRUE" *)input [11:0]a,
    (* dont_touch = "true" *)(* keep = "TRUE" *)input [11:0]b,
    (* dont_touch = "true" *)(* keep = "TRUE" *)input clk ,
    (* dont_touch = "true" *)(* keep = "TRUE" *)input rst_n ,
    (* dont_touch = "true" *)(* keep = "TRUE" *)input  start 
);
    (* dont_touch = "true" *)(* keep = "TRUE" *)reg[1:0]cnt;
//working_flag的产�?
    (* dont_touch = "true" *)(* keep = "TRUE" *)reg start_prev; // 保存前一个时刻的start信号状�??
    (* dont_touch = "true" *)(* keep = "TRUE" *)reg working_flag; // 工作标志，表示模块是否开始工�?
    always @( posedge clk or negedge rst_n) begin 
        if(!rst_n) start_prev<=0;
        else start_prev <= start; // 记录前一个时刻的start信号状�??
    end
    always @( posedge clk or negedge rst_n) begin 
        if(!rst_n) working_flag <= 0;
        else if (start == 1 && start_prev == 0) begin
            working_flag <= 1;
        end
        else if (cnt==2'b11)
            working_flag <= 0;
    end
//cnt信号
    always @( posedge clk or negedge rst_n) begin 
        if(!rst_n) cnt <= 0;
        else if (working_flag) begin
            cnt<=cnt+1'b1;
        end
    end
//求b*wn
    (* dont_touch = "true" *)(* keep = "TRUE" *)wire [11:0] wn;
    assign wn = 12'd1729;
    (* dont_touch = "true" *)(* keep = "TRUE" *)reg[11:0] product;
    // 第一�?
    (* dont_touch = "true" *)(* keep = "TRUE" *)reg [23:0] mul_reg; //  b*wn
    always @( posedge clk or negedge rst_n) begin 
        if(!rst_n) mul_reg<=0;
        else if(working_flag&&(cnt==2'b0))  mul_reg <=b*wn; // 12 bits * 12 bits -> 24 bits
        else mul_reg<=0;
    end
    //第二�?
    (* dont_touch = "true" *)(* keep = "TRUE" *)wire [3:0] mul_h , mul_m, mul_l;
    (* dont_touch = "true" *)(* keep = "TRUE" *)wire [11:0] mul_q;
    assign {mul_h, mul_m, mul_l, mul_q} = mul_reg;
    (* dont_touch = "true" *)(* keep = "TRUE" *)wire [11:0] lut_h, lut_m,lut_l;
    (* dont_touch = "true" *)(* keep = "TRUE" *)wire [12:0] add13_sum0,add13_sum1;
    (* dont_touch = "true" *)(* keep = "TRUE" *)wire [13:0] add14_sum;
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
    // 13bits 
    assign add13_sum0 = lut_h + lut_m;
    assign add13_sum1 = lut_l + mul_q;
    assign add14_sum   = add13_sum0 + add13_sum1;
    (* dont_touch = "true" *)(* keep = "TRUE" *)reg [13:0] add14_sum_reg;
    always @( posedge clk or negedge rst_n) begin 
        if(!rst_n) add14_sum_reg<=0;
        else if(working_flag&&(cnt==2'b01))  add14_sum_reg<=add14_sum;
        else    add14_sum_reg<=0;
    end
    //第三�?
    (* dont_touch = "true" *)(* keep = "TRUE" *)reg [11:0] ma0_j;
    (* dont_touch = "true" *)(* keep = "TRUE" *)wire flag;
    (* dont_touch = "true" *)(* keep = "TRUE" *)wire [11:0] sub,ma0_k,ma0_s;
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
    (* dont_touch = "true" *)(* keep = "TRUE" *)wire [11:0] CSA_cin0;
    (* dont_touch = "true" *)(* keep = "TRUE" *)wire [11:0] CSA_cin;
    (* dont_touch = "true" *)(* keep = "TRUE" *)wire [11:0] CSA_sum0;
    (* dont_touch = "true" *)(* keep = "TRUE" *)wire [11:0] s0;
    (* dont_touch = "true" *)(* keep = "TRUE" *)wire [11:0] s1;
    (* dont_touch = "true" *)(* keep = "TRUE" *)wire sel0;
    (* dont_touch = "true" *)(* keep = "TRUE" *)wire sel;
    (* dont_touch = "true" *)(* keep = "TRUE" *)wire default_sel;// useless
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
        else if(working_flag&&(cnt==2'b10))  product<=ma0_s;
        else product<=0;
    end
//求a+b*wn和a-b*wn
    (* dont_touch = "true" *)(* keep = "TRUE" *)reg [11:0]a_1;//a经过�?个寄存器之后
    (* dont_touch = "true" *)(* keep = "TRUE" *)reg [11:0]a_2;
    (* dont_touch = "true" *)(* keep = "TRUE" *)reg [11:0]a_3;
    always @( posedge clk or negedge rst_n) begin 
        if(!rst_n) a_1<=0;
        else if(working_flag&&(cnt==2'b00))  a_1 <=a;
        else  a_1<=0;
    end
    always @( posedge clk or negedge rst_n) begin 
        if(!rst_n) a_2<=0;
        else if(working_flag&&(cnt==2'b01))  a_2 <=a_1;
        else  a_2<=0;
    end  
    always @( posedge clk or negedge rst_n) begin 
        if(!rst_n) a_3<=0;
        else if(working_flag&&(cnt==2'b10))  a_3 <=a_2;
        else  a_3<=0;
    end
//第四�?
    (* dont_touch = "true" *)(* keep = "TRUE" *)reg [12:0] c;//a+b*wn
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) c<=0;
        else if(working_flag&&(a_3+product>=12'd3329)&&(cnt==2'b11))c<=a_3+product-12'd3329;
        else if(working_flag&&(cnt==2'b11)) c<=a_3+product;
        else c<=0;
    end
    (* dont_touch = "true" *)(* keep = "TRUE" *)reg [12:0] d;//a-b*wn
    always @(posedge clk or negedge rst_n) begin
        if(!rst_n) d<=0;
        else if(working_flag&&(cnt==2'b11)&&(a_3>=product))   d<=a_3-product;
        else if(working_flag&&(cnt==2'b11)) d<=12'd3329-product+a_3;
        else d<=0;
    end

//     ila_0 the_ila (
//.clk(clk),


//.probe0(start),
//.probe1(cnt),
//.probe2(a),
//.probe3(b),
//.probe4(start_prev),
//.probe5(working_flag),
//.probe6(c),
//.probe7(d)

//);

endmodule





module a_b_mul_tb;

    reg clk,rst_n,start;
    reg [11:0] a,b;
    a_b_mul a_b_mul_inst(
        .start ( start ),
        .a ( a ),
        .b  (b),
        .clk   ( clk   ),
        .rst_n  ( rst_n  )
    );

    always#5 clk = ~clk;

    initial begin
        clk = 0;
        rst_n = 1;
        start =0;
        #10 rst_n = 0;
        #10 rst_n = 1;
        #50 start=1;
        #10 start=0;
        #200 start=1;
    end

    always@(posedge clk or negedge rst_n) begin
        if(!rst_n) a <= 12'd567;

    end
    always@(posedge clk or negedge rst_n) begin
        if(!rst_n) b <= 12'd1000;

    end




endmodule