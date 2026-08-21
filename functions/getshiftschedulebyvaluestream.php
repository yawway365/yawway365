<?php
function GetShiftScheduleByValueStream($user) {

    $return_arr = array();

    if(isset($_GET['valueStream'])) {
        $value_stream = $_GET['valueStream'];
    }//Closes IF
    else if(isset($_POST['valueStream'])) {
        $value_stream = $_POST['valueStream'];
    }//Closes ELSE IF
    else {
        $return_arr['ERROR'] = 'valueStream was not provided';
        return json_encode($return_arr);
    }//Closes ELSE

    if(isset($_GET['startDate'])) {
        $start_date = $_GET['startDate'];
    }//Closes IF
    else if(isset($_POST['startDate'])) {
        $start_date = $_POST['startDate'];
    }//Closes ELSE IF
    else {
        $return_arr['ERROR'] = 'startDate was not provided';
        return json_encode($return_arr);
    }//Closes ELSE

    if(isset($_GET['endDate'])) {
        $end_date = $_GET['endDate'];
    }//Closes IF
    else if(isset($_POST['endDate'])) {
        $end_date = $_POST['endDate'];
    }//Closes ELSE IF
    else {
        $return_arr['ERROR'] = 'endDate was not provided';
        return json_encode($return_arr);
    }//Closes ELSE

    //STEP 1 - Production DB - resolve the value stream down to its workareas.
    //This result set is fully drained before we touch the Scheduling DB, because
    //mysql_connect() hands back the SAME link when the host/user/pass match, and
    //SchedulingConnect() would then switch the active database out from under us.
    $conn = ProductionConnect();

    $workarea_arr  = array();
    $workarea_list = array();

    $sql = "SELECT
                workareas.WorkAreaID,
                workareas.WorkArea,
                workareas.WorkAreaDescription,
                departments.DepartmentID,
                departments.Department,
                departments.DepartmentName,
                departments.Plant,
                departments.ValueStream
            FROM workareas
            INNER JOIN departments ON departments.DepartmentID = workareas.DepartmentID
            WHERE departments.ValueStream = '" . mysql_real_escape_string($value_stream, $conn) . "'
            ORDER BY departments.Department, workareas.WorkArea";

    $result = mysql_query($sql, $conn);
    while($row = mysql_fetch_assoc($result)) {
        $row['shifts'] = array();
        $workarea_arr[$row['WorkArea']] = $row;
        $workarea_list[] = "'" . mysql_real_escape_string($row['WorkArea'], $conn) . "'";
    }//Closes WHILE

    $count = count($workarea_arr);
    if($count == 0) {
        $return_arr['ERROR'] = "No WorkAreas found for the value stream '$value_stream'";
        return json_encode($return_arr);
    }//Closes IF

    //STEP 2 - Scheduling DB - ONE query for every workarea in the value stream.
    $conn = SchedulingConnect();

    $sql = "SELECT
                WorkArea,
                ProductionDate,
                FirstShift,
                FirstShiftEnd,
                FirstShiftEnabled,
                SecondShift,
                SecondShiftEnd,
                SecondShiftEnabled,
                ThirdShift,
                ThirdShiftEnd,
                ThirdShiftEnabled
            FROM shiftsbyworkarea
            WHERE WorkArea IN (" . implode(',', $workarea_list) . ")
            AND ProductionDate BETWEEN '" . mysql_real_escape_string($start_date, $conn) . "'
                                 AND '" . mysql_real_escape_string($end_date, $conn) . "'
            ORDER BY WorkArea, ProductionDate";

    $shift_count = 0;
    $result = mysql_query($sql, $conn);
    while($row = mysql_fetch_assoc($result)) {
        //Bucket each row under its workarea instead of running a query per workarea
        if(isset($workarea_arr[$row['WorkArea']])) {
            array_push($workarea_arr[$row['WorkArea']]['shifts'], $row);
            $shift_count++;
        }//Closes IF
    }//Closes WHILE

    //Workareas with no rows in the range stay in the response with an empty
    //shifts array so the UI can still draw the row and show the gap.
    $final_arr = array();
    $final_arr['ValueStream'] = $value_stream;
    $final_arr['StartDate']   = $start_date;
    $final_arr['EndDate']     = $end_date;
    $final_arr['WorkAreas']   = $workarea_arr;

    if($shift_count == 0) {
        $final_arr['NOTE'] = "No shift schedule available for the date range '$start_date' to '$end_date'";
    }//Closes IF

    return json_encode($final_arr);

}//Closes FUNCTION
?>
