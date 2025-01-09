module subrotinas
    implicit none
    
contains
    subroutine relative_vector(x1, y1, x2, y2, dx, dy)
        real, intent(in) :: x1, y1, x2, y2
        real, intent(out) ::  dx, dy

        dx = x2 - x1
        dy = y2 - y1
        
    end subroutine relative_vector

    subroutine spring_force(dx, dy, natural_distance, k_mol, fx, fy)
        implicit none
        real, intent(in) :: dx, dy, natural_distance, k_mol
        real :: particle_distance, delta_distance, force
        real, intent(out) ::  fx, fy

        particle_distance = sqrt(dx**2 + dy**2)
        delta_distance = particle_distance - natural_distance
        
        if ( particle_distance == 0 ) then
            fx = 0
            fy = 0
        end if
        ! Calculate force
        force = -k_mol * delta_distance
        ! Normalized directional vector
        fx = force * (dx / particle_distance)
        fy = force * (dx / particle_distance)

    end subroutine spring_force

    subroutine damped_string_acc(vx, vy, fx, fy, b, mass, ax, ay)
        implicit none
        ! Entradas
        real, dimension(:), intent(in) :: vx, vy, mass
        real, intent(in) :: fx, fy, b
        real, dimension(:), intent(out) :: ax, ay
        
        real, dimension(size(vx)) :: f_damps_x, f_damps_y
        integer :: i, n

        n = size(vx)
    
        do i = 1, n
            ! Damping forces
            f_damps_x(i) = b * vx(i)
            f_damps_y(i) = b * vy(i)
        end do
    
        do i = 1, n
            ! Calculate acceleration for each particle
            if (i == 1) then
                ax(i) = (fx - f_damps_x(i)) / mass(i)
                ay(i) = (fy - f_damps_y(i)) / mass(i)
            else
                ax(i) = -(fx - f_damps_x(i)) / mass(i)
                ay(i) = -(fy - f_damps_y(i)) / mass(i)
            end if
        end do
    end subroutine damped_string_acc
        
end module subrotinas
