
function M=getransformation_matrix(par)

switch(length(par))
    case 6  %3d
        M=make_transformation_matrix(par(1:3),par(4:6));
    case 9  %3d
        M=make_transformation_matrix(par(1:3),par(4:6),par(7:9));
    case 15 %3d
        M=make_transformation_matrix(par(1:3),par(4:6),par(7:9),par(10:15));
    case 3 % 2d
        M=make_transformation_matrix(par(1:2),par(3));
    case 5 % 2d
        M=make_transformation_matrix(par(1:2),par(3),par(4:5));
    case 7 % 2d
        M=make_transformation_matrix(par(1:2),par(3),par(4:5),par(6:7));
        
end