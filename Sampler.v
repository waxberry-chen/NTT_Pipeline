module Sampler (
    (* dont_touch = "true" *)(* keep = "TRUE" *) input wire Start,
    (* dont_touch = "true" *)(* keep = "TRUE" *) input wire[5:0]s,
    (* dont_touch = "true" *)(* keep = "TRUE" *) input wire clk,rst_n
);
    

    (* dont_touch = "true" *)(* keep = "TRUE" *)reg [7:0] s_reg; 

    always@(posedge clk or negedge rst_n) begin
        if(!rst_n) s_reg <= 6'd0;
        else if(Start) s_reg <= s;
    end

    (* dont_touch = "true" *)(* keep = "TRUE" *)wire [11:0] s_coeff;
    (* dont_touch = "true" *)(* keep = "TRUE" *)wire [ 1:0] a,b;
    assign a = s_reg[0] + s_reg[1] + s_reg[2];
    assign b = s_reg[3] + s_reg[4] + s_reg[5];

    assign s_coeff = a > b ? a - b : a - b + 12'd3329;


    (* dont_touch = "true" *)(* keep = "TRUE" *)reg Start_reg;
    always@(posedge clk or negedge rst_n) begin
        if(!rst_n) Start_reg <= 1'b0;
        else Start_reg <= Start;
    end



dpram#(
   .WIDTH              ( 12 ),
   .DEPTH              ( 1024 ),
   .Sel                ( 0 )
)uRAM(
   .clk                ( clk                ),
   .ena                ( Start_reg          ),
   .enb                ( 1'b0               ),
   .wea                ( Start_reg          ),
   .web                ( 1'b0               ),
   .addra              ( 10'd0              ),
   .addrb              ( 10'd0              ),
   .dina               ( s_coeff               ),
   .dinb               ( 12'd0               ),
   .douta              (               ),
   .doutb              (               )
);


endmodule

module Sampler_tb;

    reg clk,rst_n,Start;
    reg [5:0] s;
    Sampler u_Sampler(
        .Start ( Start ),
        .s ( s ),
        .clk   ( clk   ),
        .rst_n  ( rst_n  )
    );

    always#5 clk = ~clk;

    initial begin
        clk = 0;
        rst_n = 1;
        //s = 0;
        #10 rst_n = 0;
        #10 rst_n = 1;

    end

    always@(posedge clk or negedge rst_n) begin
        if(!rst_n) s <= 5'd0;
        else if(Start) s <= s + 1'b1;
    end

    always@(posedge clk or negedge rst_n) begin 
        if(!rst_n) Start <= 1'b1;
        else Start <= ~Start;
    end


endmodule