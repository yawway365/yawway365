<?php
function GetValueStreams($admin_auth, $user) {

    $conn = ProductionConnect();
    $return_arr = array();

    //Distinct value streams that actually have workareas attached to them
    $sql = "SELECT
                departments.ValueStream,
                COUNT(workareas.WorkAreaID) AS WorkAreaCount
            FROM workareas
            INNER JOIN departments ON departments.DepartmentID = workareas.DepartmentID
            WHERE departments.ValueStream != ''
            GROUP BY departments.ValueStream
            ORDER BY departments.ValueStream";

    $result = mysql_query($sql, $conn);
    while($row = mysql_fetch_assoc($result)) {
        $return_arr[$row['ValueStream']] = $row;
    }//Closes WHILE

    $count = count($return_arr);
    if($count > 0) {
        return json_encode($return_arr);
    }//Closes IF
    else {
        $return_arr['NOTE'] = 'No ValueStreams Found';
        return json_encode($return_arr);
    }//Closes ELSE

}//Closes FUNCTION
?>
