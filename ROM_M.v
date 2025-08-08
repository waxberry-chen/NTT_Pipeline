


module ROM_M (
input [3:0]	a,
output reg [11:0] spo
);

always@(*)
begin
	case(a)
	4'd0: spo = 12'd0;
	4'd1: spo = 12'd2285;
	4'd2: spo = 12'd1241;
	4'd3: spo = 12'd197;
	4'd4: spo = 12'd2482;
	4'd5: spo = 12'd1438;
	4'd6: spo = 12'd394;
	4'd7: spo = 12'd2679;
	4'd8: spo = 12'd1635;
	4'd9: spo = 12'd591;
	4'd10: spo = 12'd2876;
	4'd11: spo = 12'd1832;
	4'd12: spo = 12'd788;
	4'd13: spo = 12'd3073;	
	4'd14: spo = 12'd2029;
	4'd15: spo = 12'd985;		
	endcase
end

// ROM_H rom0 (
// .a (mul_h),
// .spo (lut_h)
// );

endmodule